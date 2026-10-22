#!/usr/bin/env bash
# Manage one isolated Argus validation runtime on gpu-validation.
#
# Signals are sent only to a process group whose boot id, leader start time,
# PID/PGID/SID, command line, working directory, and inherited owner token all
# match the metadata written by this script.  This deliberately has no
# process-name or port-owner kill fallback.

set -Eeuo pipefail
umask 077

RUNTIME_HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
[[ -n "$RUNTIME_HOME" && "$RUNTIME_HOME" == /* ]] || {
  echo "Cannot resolve the runtime account home directory" >&2
  exit 2
}
readonly RUNTIME_HOME
readonly PRODUCTION_CHECKOUT="$(realpath -- "$RUNTIME_HOME/argus")"
readonly PROCESS_METADATA_VERSION="1"

STARTED_THIS_INVOCATION=0

usage() {
  cat <<'EOF'
Usage: gpu_validation_runtime.sh <start|stop|restart|status|health|diagnose> --env FILE

Required env-file variables:
  ARGUS_ROOT                isolated, immutable source snapshot
  ARGUS_STATE_DIR           independent runtime state outside the source tree
  ARGUS_CONFIG              validation-only Argus YAML
  ARGUS_PYTHON              Python 3.11 interpreter (the shared .venv is allowed)
  GO2RTC_BIN                Linux go2rtc executable copied into validation state
  GO2RTC_CONFIG_DIR         directory where Argus writes go2rtc.json
  SOURCE_SHA256_MANIFEST    strict source inventory rooted at ARGUS_ROOT
  ARTIFACT_SHA256_MANIFEST  sha256sum-compatible manifest rooted at ARGUS_STATE_DIR
  ARGUS_API_TOKEN_FILE      0600 API-token file below ARGUS_STATE_DIR
  ARGUS_API_TOKEN           token sealed in validation.env; must match the file
  CAMERA_ID                 validation camera id
  MODEL_VERSION_ID          exact registry/runtime model version
  MODEL_STAGE               exact registry stage (must be canary)
  RUNTIME_ARTIFACT_SHA256   complete runtime-artifact digest
  SCENE_PROFILE_ID          exact active SceneGuard profile
  RELEASE_REVISION          exact registry release revision

Optional variables:
  ARGUS_PORT=18081
  GO2RTC_API_PORT=11984
  GO2RTC_RTSP_PORT=18554
  GO2RTC_WEBRTC_PORT=18555
  CAMERA_DEVICE=/dev/video0
  STARTUP_TIMEOUT_SECONDS=120
EOF
}

fail() {
  echo "$*" >&2
  return 1
}

is_within() {
  local path=$1
  local root=$2
  [[ "$path" == "$root" || "$path" == "$root"/* ]]
}

validate_port() {
  local name=$1
  local value=$2
  [[ "$value" =~ ^[0-9]{1,5}$ ]] || {
    fail "$name must be an integer"
    return 1
  }
  local numeric=$((10#$value))
  (( numeric >= 1024 && numeric <= 65535 )) || {
    fail "$name must be between 1024 and 65535"
    return 1
  }
}

[[ $# -ge 1 ]] || { usage; exit 2; }
COMMAND=$1
shift
ENV_FILE=${ARGUS_VALIDATION_ENV:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      [[ $# -ge 2 ]] || { echo "--env requires a file" >&2; exit 2; }
      ENV_FILE=$2
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$ENV_FILE" && -f "$ENV_FILE" && ! -L "$ENV_FILE" ]] || {
  echo "A readable, regular --env FILE is required" >&2
  exit 2
}
ENV_FILE=$(realpath -- "$ENV_FILE")
[[ "$(basename -- "$ENV_FILE")" == "validation.env" ]] || {
  echo "The lifecycle anchor must be a file named validation.env" >&2
  exit 2
}
ENV_ANCHOR_STATE=$(realpath -- "$(dirname -- "$ENV_FILE")")
anchor_uid=$(stat -c '%u' -- "$ENV_ANCHOR_STATE")
anchor_mode=$(stat -c '%a' -- "$ENV_ANCHOR_STATE")
env_uid=$(stat -c '%u' -- "$ENV_FILE")
env_mode=$(stat -c '%a' -- "$ENV_FILE")
[[ "$anchor_uid" == "$(id -u)" && "$anchor_mode" == "700" ]] || {
  echo "The validation state anchor must be owned by the runtime user with mode 0700" >&2
  exit 2
}
[[ "$env_uid" == "$(id -u)" && "$env_mode" == "600" ]] || {
  echo "validation.env must be owned by the runtime user with mode 0600" >&2
  exit 2
}
readonly COMMAND ENV_FILE ENV_ANCHOR_STATE
# shellcheck disable=SC1090
source "$ENV_FILE"

# A caller shell or service manager must not redirect storage/auth through an
# ambient nested override. Controlled overrides are added to the child below.
while IFS='=' read -r name _; do
  case "$name" in
    ARGUS__*) unset "$name" ;;
  esac
done < <(env)

required=(
  ARGUS_ROOT ARGUS_STATE_DIR ARGUS_CONFIG ARGUS_PYTHON GO2RTC_BIN GO2RTC_CONFIG_DIR
  SOURCE_SHA256_MANIFEST ARTIFACT_SHA256_MANIFEST ARGUS_API_TOKEN_FILE ARGUS_API_TOKEN
  CAMERA_ID
  MODEL_VERSION_ID MODEL_STAGE RUNTIME_ARTIFACT_SHA256 SCENE_PROFILE_ID
  RELEASE_REVISION
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing required variable: $name" >&2; exit 2; }
done

ARGUS_PORT=${ARGUS_PORT:-18081}
GO2RTC_API_PORT=${GO2RTC_API_PORT:-11984}
GO2RTC_RTSP_PORT=${GO2RTC_RTSP_PORT:-18554}
GO2RTC_WEBRTC_PORT=${GO2RTC_WEBRTC_PORT:-18555}
CAMERA_DEVICE=${CAMERA_DEVICE:-/dev/video0}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-120}

for port_name in ARGUS_PORT GO2RTC_API_PORT GO2RTC_RTSP_PORT GO2RTC_WEBRTC_PORT; do
  validate_port "$port_name" "${!port_name}" || exit 2
  printf -v "$port_name" '%d' "$((10#${!port_name}))"
done
[[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[0-9]{1,3}$ ]] || {
  echo "STARTUP_TIMEOUT_SECONDS must be an integer between 10 and 900" >&2
  exit 2
}
STARTUP_TIMEOUT_SECONDS=$((10#$STARTUP_TIMEOUT_SECONDS))
(( STARTUP_TIMEOUT_SECONDS >= 10 && STARTUP_TIMEOUT_SECONDS <= 900 )) || {
    echo "STARTUP_TIMEOUT_SECONDS must be an integer between 10 and 900" >&2
    exit 2
  }
declare -A _ports_seen=()
for port in "$ARGUS_PORT" "$GO2RTC_API_PORT" "$GO2RTC_RTSP_PORT" "$GO2RTC_WEBRTC_PORT"; do
  [[ -z "${_ports_seen[$port]+present}" ]] || {
    echo "Validation TCP/UDP ports must be distinct (duplicate: $port)" >&2
    exit 2
  }
  _ports_seen[$port]=1
done

[[ "$CAMERA_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || {
  echo "CAMERA_ID has an unsafe value" >&2
  exit 2
}
[[ "$SCENE_PROFILE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || {
  echo "SCENE_PROFILE_ID has an unsafe value" >&2
  exit 2
}
[[ "$MODEL_VERSION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "MODEL_VERSION_ID has an unsafe value" >&2
  exit 2
}
[[ "$MODEL_STAGE" == "canary" ]] || {
  echo "Isolated gpu-validation validation must remain at canary, never production" >&2
  exit 2
}
[[ "$RUNTIME_ARTIFACT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "RUNTIME_ARTIFACT_SHA256 must contain 64 hex digits" >&2
  exit 2
}
RUNTIME_ARTIFACT_SHA256=${RUNTIME_ARTIFACT_SHA256,,}
[[ "$RELEASE_REVISION" =~ ^[0-9]{1,10}$ ]] || {
  echo "RELEASE_REVISION must be a non-negative integer" >&2
  exit 2
}
RELEASE_REVISION=$((10#$RELEASE_REVISION))

ARGUS_ROOT=$(realpath -- "$ARGUS_ROOT")
ARGUS_STATE_DIR=$(realpath -- "$ARGUS_STATE_DIR")
ARGUS_CONFIG=$(realpath -- "$ARGUS_CONFIG")
# Preserve the venv entrypoint path instead of dereferencing it to the base
# interpreter; invoking that path is what gives Python the venv sys.prefix.
ARGUS_PYTHON="$(cd "$(dirname "$ARGUS_PYTHON")" && pwd -P)/$(basename "$ARGUS_PYTHON")"
GO2RTC_BIN=$(realpath -- "$GO2RTC_BIN")
GO2RTC_CONFIG_DIR=$(realpath -m -- "$GO2RTC_CONFIG_DIR")
SOURCE_SHA256_MANIFEST=$(realpath -- "$SOURCE_SHA256_MANIFEST")
ARTIFACT_SHA256_MANIFEST=$(realpath -- "$ARTIFACT_SHA256_MANIFEST")
ARGUS_API_TOKEN_FILE=$(realpath -- "$ARGUS_API_TOKEN_FILE")
CAMERA_DEVICE=$(realpath -m -- "$CAMERA_DEVICE")

[[ -d "$ARGUS_ROOT" && -d "$ARGUS_STATE_DIR" ]] || {
  echo "ARGUS_ROOT and ARGUS_STATE_DIR must already exist as directories" >&2
  exit 2
}
[[ -x "$ARGUS_PYTHON" && -x "$GO2RTC_BIN" ]] || {
  echo "ARGUS_PYTHON and GO2RTC_BIN must be executable" >&2
  exit 2
}
[[ "$ARGUS_STATE_DIR" == "$ENV_ANCHOR_STATE" ]] || {
  echo "ARGUS_STATE_DIR must equal the directory anchored by --env" >&2
  exit 2
}

# This boundary is fixed in the script. It is not caller-overridable.
if is_within "$ARGUS_ROOT" "$PRODUCTION_CHECKOUT"; then
  echo "ARGUS_ROOT must not be the runtime account production checkout or any path below it" >&2
  exit 2
fi
if is_within "$ARGUS_STATE_DIR" "$PRODUCTION_CHECKOUT"; then
  echo "ARGUS_STATE_DIR must not be the runtime account production checkout or any path below it" >&2
  exit 2
fi
[[ "$ARGUS_STATE_DIR" != "$ARGUS_ROOT" ]] \
  && ! is_within "$ARGUS_STATE_DIR" "$ARGUS_ROOT" \
  && ! is_within "$ARGUS_ROOT" "$ARGUS_STATE_DIR" || {
    echo "Source snapshot and validation state must be disjoint" >&2
    exit 2
  }
for owned_path in \
  "$ENV_FILE" "$ARGUS_CONFIG" "$GO2RTC_BIN" "$GO2RTC_CONFIG_DIR" \
  "$SOURCE_SHA256_MANIFEST" "$ARTIFACT_SHA256_MANIFEST" "$ARGUS_API_TOKEN_FILE"; do
  is_within "$owned_path" "$ARGUS_STATE_DIR" || {
    echo "Validation-owned path escaped ARGUS_STATE_DIR: $owned_path" >&2
    exit 2
  }
done

declare -A fixed_state_paths=(
  [ENV_FILE]="$ARGUS_STATE_DIR/validation.env"
  [ARGUS_CONFIG]="$ARGUS_STATE_DIR/argus-validation.yaml"
  [GO2RTC_BIN]="$ARGUS_STATE_DIR/bin/go2rtc"
  [GO2RTC_CONFIG_DIR]="$ARGUS_STATE_DIR/go2rtc"
  [SOURCE_SHA256_MANIFEST]="$ARGUS_STATE_DIR/source.sha256"
  [ARTIFACT_SHA256_MANIFEST]="$ARGUS_STATE_DIR/artifacts.sha256"
  [ARGUS_API_TOKEN_FILE]="$ARGUS_STATE_DIR/api-token"
)
for fixed_name in "${!fixed_state_paths[@]}"; do
  actual_value=${!fixed_name}
  expected_value=${fixed_state_paths[$fixed_name]}
  [[ "$actual_value" == "$expected_value" ]] || {
    echo "$fixed_name must be fixed at $expected_value (got $actual_value)" >&2
    exit 2
  }
done

token_uid=$(stat -c '%u' -- "$ARGUS_API_TOKEN_FILE")
token_mode=$(stat -c '%a' -- "$ARGUS_API_TOKEN_FILE")
[[ "$token_uid" == "$(id -u)" && "$token_mode" == "600" ]] || {
  echo "ARGUS_API_TOKEN_FILE must be owned by the runtime user with mode 0600" >&2
  exit 2
}
mapfile -t _token_lines < "$ARGUS_API_TOKEN_FILE"
[[ ${#_token_lines[@]} -eq 1 && -n "${_token_lines[0]}" ]] || {
  echo "ARGUS_API_TOKEN_FILE must contain exactly one non-empty line" >&2
  exit 2
}
[[ "${_token_lines[0]}" == "$ARGUS_API_TOKEN" ]] || {
  echo "ARGUS_API_TOKEN_FILE does not match the token sealed in validation.env" >&2
  exit 2
}
[[ "${_token_lines[0]}" =~ ^[A-Za-z0-9_-]{32,128}$ ]] || {
  echo "ARGUS_API_TOKEN_FILE contains an invalid token format" >&2
  exit 2
}
unset API_TOKEN 2>/dev/null || true
readonly API_TOKEN=${_token_lines[0]}
unset \
  _token_lines ARGUS_API_TOKEN \
  ARGUS_E2E_PASSWORD ARGUS_E2E_VIEWER_PASSWORD \
  ARGUS_E2E_USERNAME ARGUS_E2E_VIEWER_USERNAME \
  2>/dev/null || true

PID_DIR="$ARGUS_STATE_DIR/pids"
LOG_DIR="$ARGUS_STATE_DIR/logs"
ARGUS_PROCESS_METADATA="$PID_DIR/argus.process"
LEGACY_PID_FILE="$PID_DIR/argus.pid"
ARGUS_LOG="$LOG_DIR/argus.log"
GO2RTC_GENERATED_CONFIG="$GO2RTC_CONFIG_DIR/go2rtc.json"
RUNTIME_LOCK_FILE="$ARGUS_STATE_DIR/.gpu-validation-validation-runtime.lock"
mkdir -p "$PID_DIR" "$LOG_DIR" "$GO2RTC_CONFIG_DIR"
mkdir -p \
  "$ARGUS_STATE_DIR/cache/xdg" \
  "$ARGUS_STATE_DIR/cache/torch" \
  "$ARGUS_STATE_DIR/cache/ultralytics" \
  "$ARGUS_STATE_DIR/cache/matplotlib" \
  "$ARGUS_STATE_DIR/cache/huggingface" \
  "$ARGUS_STATE_DIR/pycache"

command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }
command -v setsid >/dev/null 2>&1 || { echo "setsid is required" >&2; exit 2; }
command -v ss >/dev/null 2>&1 || { echo "ss is required" >&2; exit 2; }
exec 9>"$RUNTIME_LOCK_FILE"
flock -n 9 || {
  echo "Another validation lifecycle command holds $RUNTIME_LOCK_FILE" >&2
  exit 75
}

curl_api() {
  local path=$1
  # Feed the header through stdin so the secret never appears in curl's
  # process command line or the runtime log.
  curl --silent --show-error --fail --max-time 10 --config - \
    "http://127.0.0.1:${ARGUS_PORT}${path}" <<EOF
header = "X-API-Token: $API_TOKEN"
EOF
}

wait_http() {
  local url=$1
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until curl --silent --show-error --fail --max-time 3 "$url" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
}

wait_api() {
  local path=$1
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until curl_api "$path" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
}

verify_manifest() {
  local root=$1
  local manifest=$2
  local exact=$3
  "$ARGUS_PYTHON" -I -S - "$root" "$manifest" "$exact" <<'PY'
from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath


root = Path(sys.argv[1]).resolve(strict=True)
manifest = Path(sys.argv[2]).resolve(strict=True)
exact = sys.argv[3] == "1"
try:
    lines = manifest.read_text(encoding="utf-8").splitlines()
except (OSError, UnicodeError) as exc:
    raise SystemExit(f"manifest is unreadable: {exc}") from exc
if not lines:
    raise SystemExit("manifest is empty")

expected: dict[str, str] = {}
line_re = re.compile(r"^([0-9A-Fa-f]{64}) ([ *])(.+)$")
for number, line in enumerate(lines, 1):
    if line.startswith("\\"):
        raise SystemExit(f"manifest line {number} uses an escaped filename")
    match = line_re.fullmatch(line)
    if match is None:
        raise SystemExit(f"manifest line {number} is not sha256sum-compatible")
    raw_name = match.group(3)
    if raw_name.startswith("./"):
        raw_name = raw_name[2:]
    rel = PurePosixPath(raw_name)
    if (
        not raw_name
        or rel.is_absolute()
        or raw_name != rel.as_posix()
        or any(part in {"", ".", ".."} for part in rel.parts)
        or "\\" in raw_name
    ):
        raise SystemExit(f"manifest line {number} has an unsafe path: {raw_name!r}")
    if raw_name in expected:
        raise SystemExit(f"manifest contains duplicate path: {raw_name}")
    candidate = root.joinpath(*rel.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"manifest path escaped or is missing: {raw_name}: {exc}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise SystemExit(f"manifest path is not a regular non-symlink file: {raw_name}")
    expected[raw_name] = match.group(1).lower()

if exact:
    actual: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        for dirname in dirnames:
            path = directory / dirname
            if path.is_symlink():
                raise SystemExit(
                    f"source snapshot contains an unhashable directory symlink: "
                    f"{path.relative_to(root).as_posix()}"
                )
        for filename in filenames:
            path = directory / filename
            rel_name = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise SystemExit(f"source snapshot contains a file symlink: {rel_name}")
            try:
                mode = path.stat().st_mode
            except OSError as exc:
                raise SystemExit(f"source file is unreadable: {rel_name}: {exc}") from exc
            if not stat.S_ISREG(mode):
                raise SystemExit(f"source snapshot contains a special file: {rel_name}")
            actual.add(rel_name)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:20]))
        if extra:
            details.append("unmanifested=" + ", ".join(extra[:20]))
        raise SystemExit("strict manifest file-set mismatch: " + "; ".join(details))

for rel_name, wanted in sorted(expected.items()):
    digest = hashlib.sha256()
    with (root / rel_name).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    actual_digest = digest.hexdigest()
    if not hmac.compare_digest(actual_digest, wanted):
        raise SystemExit(
            f"SHA256 mismatch for {rel_name}: expected {wanted}, got {actual_digest}"
        )
PY
}

verify_source() {
  verify_manifest "$ARGUS_ROOT" "$SOURCE_SHA256_MANIFEST" 1
}

verify_artifacts() {
  verify_manifest "$ARGUS_STATE_DIR" "$ARTIFACT_SHA256_MANIFEST" 0
}

socket_table() {
  local protocol=$1
  case "$protocol" in
    tcp) ss -H -ltn ;;
    udp) ss -H -lun ;;
    *) return 2 ;;
  esac
}

socket_endpoints_for_port() {
  local protocol=$1
  local port=$2
  local state recvq sendq endpoint peer remainder
  while read -r state recvq sendq endpoint peer remainder; do
    [[ "${endpoint##*:}" == "$port" ]] || continue
    printf '%s\n' "$endpoint"
  done < <(socket_table "$protocol")
}

assert_ports_free() {
  local protocol port endpoints
  for protocol in tcp udp; do
    for port in \
      "$ARGUS_PORT" "$GO2RTC_API_PORT" "$GO2RTC_RTSP_PORT" "$GO2RTC_WEBRTC_PORT"; do
      endpoints=$(socket_endpoints_for_port "$protocol" "$port")
      [[ -z "$endpoints" ]] || {
        echo "Preflight failed: $protocol port $port is already bound at: $endpoints" >&2
        return 1
      }
    done
  done
}

assert_loopback_listener() {
  local protocol=$1
  local port=$2
  local endpoints endpoint host found=0
  endpoints=$(socket_endpoints_for_port "$protocol" "$port")
  [[ -n "$endpoints" ]] || {
    echo "FAIL missing $protocol listener on port $port" >&2
    return 1
  }
  while IFS= read -r endpoint; do
    [[ -n "$endpoint" ]] || continue
    found=1
    host=${endpoint%:*}
    host=${host#[}
    host=${host%]}
    [[ "$host" == "127.0.0.1" ]] || {
      echo "FAIL $protocol port $port is not loopback-only: $endpoint" >&2
      return 1
    }
  done <<< "$endpoints"
  [[ "$found" == 1 ]]
}

read_proc_identity() {
  local pid=$1
  local stat_line rest
  local -a fields=()
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/stat" ]] || return 1
  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  rest=${stat_line##*) }
  read -r -a fields <<< "$rest"
  (( ${#fields[@]} >= 20 )) || return 1
  PROC_STATE=${fields[0]}
  PROC_PPID=${fields[1]}
  PROC_PGID=${fields[2]}
  PROC_SID=${fields[3]}
  PROC_STARTTIME=${fields[19]}
}

load_process_metadata() {
  [[ -f "$ARGUS_PROCESS_METADATA" && ! -L "$ARGUS_PROCESS_METADATA" ]] || return 1
  [[ "$(stat -c '%u' -- "$ARGUS_PROCESS_METADATA")" == "$(id -u)" \
    && "$(stat -c '%a' -- "$ARGUS_PROCESS_METADATA")" == "600" ]] || return 1
  local key value
  declare -A values=()
  while IFS='=' read -r key value; do
    case "$key" in
      version|boot_id|pid|starttime|pgid|sid|owner_token) ;;
      *) return 1 ;;
    esac
    [[ -z "${values[$key]+present}" ]] || return 1
    values[$key]=$value
  done < "$ARGUS_PROCESS_METADATA"
  [[ ${#values[@]} -eq 7 ]] || return 1
  [[ "${values[version]}" == "$PROCESS_METADATA_VERSION" ]] || return 1
  [[ "${values[boot_id]}" =~ ^[0-9a-fA-F-]{36}$ ]] || return 1
  for key in pid starttime pgid sid; do
    [[ "${values[$key]}" =~ ^[0-9]+$ ]] || return 1
  done
  [[ "${values[owner_token]}" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "${values[pid]}" == "${values[pgid]}" \
    && "${values[pid]}" == "${values[sid]}" ]] || return 1
  META_BOOT_ID=${values[boot_id]}
  META_PID=${values[pid]}
  META_STARTTIME=${values[starttime]}
  META_PGID=${values[pgid]}
  META_SID=${values[sid]}
  META_OWNER_TOKEN=${values[owner_token]}
}

process_environment_has() {
  local pid=$1
  local wanted=$2
  local entry
  [[ -r "/proc/$pid/environ" ]] || return 1
  while IFS= read -r -d '' entry; do
    [[ "$entry" == "$wanted" ]] && return 0
  done < "/proc/$pid/environ"
  return 1
}

argus_cmdline_matches() {
  local pid=$1
  local expected_exe actual_exe
  local -a args=()
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  mapfile -d '' -t args < "/proc/$pid/cmdline"
  [[ ${#args[@]} -eq 5 \
    && "${args[1]:-}" == "-m" \
    && "${args[2]:-}" == "argus" \
    && "${args[3]:-}" == "--config" \
    && "${args[4]:-}" == "$ARGUS_CONFIG" ]] || return 1
  expected_exe=$(realpath -- "$ARGUS_PYTHON")
  actual_exe=$(realpath -- "/proc/$pid/exe") || return 1
  [[ "$actual_exe" == "$expected_exe" ]]
}

process_matches_metadata() {
  load_process_metadata || return 1
  [[ "$(< /proc/sys/kernel/random/boot_id)" == "$META_BOOT_ID" ]] || return 1
  read_proc_identity "$META_PID" || return 1
  [[ "$PROC_STATE" != "Z" \
    && "$PROC_STARTTIME" == "$META_STARTTIME" \
    && "$PROC_PGID" == "$META_PGID" \
    && "$PROC_SID" == "$META_SID" ]] || return 1
  argus_cmdline_matches "$META_PID" || return 1
  [[ "$(readlink -f -- "/proc/$META_PID/cwd")" == "$ARGUS_STATE_DIR" ]] || return 1
  process_environment_has "$META_PID" \
    "ARGUS_VALIDATION_OWNER_TOKEN=$META_OWNER_TOKEN" || return 1
  process_environment_has "$META_PID" "PYTHONPATH=$ARGUS_ROOT/src" || return 1
  process_environment_has "$META_PID" "ARGUS__DASHBOARD__HOST=127.0.0.1" || return 1
  process_environment_has "$META_PID" "ARGUS__DASHBOARD__PORT=$ARGUS_PORT" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_CONFIG_DIR=$GO2RTC_CONFIG_DIR" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_API_PORT=$GO2RTC_API_PORT" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_RTSP_PORT=$GO2RTC_RTSP_PORT" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_WEBRTC_PORT=$GO2RTC_WEBRTC_PORT" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_WEBRTC_LISTEN_HOST=127.0.0.1" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_BINARY=$GO2RTC_BIN" || return 1
  process_environment_has "$META_PID" \
    "ARGUS__DASHBOARD__GO2RTC_REPLACE_STALE_PROCESS=false" || return 1
}

OWNED_GROUP_MEMBERS=()
collect_owned_group_members() {
  load_process_metadata || {
    echo "Invalid or missing process metadata: $ARGUS_PROCESS_METADATA" >&2
    return 1
  }
  [[ "$(< /proc/sys/kernel/random/boot_id)" == "$META_BOOT_ID" ]] || {
    echo "Process metadata belongs to a different boot; refusing signals" >&2
    return 1
  }
  OWNED_GROUP_MEMBERS=()
  local proc pid
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    read_proc_identity "$pid" || continue
    [[ "$PROC_STATE" != "Z" \
      && "$PROC_PGID" == "$META_PGID" \
      && "$PROC_SID" == "$META_SID" ]] || continue
    if ! process_environment_has "$pid" \
      "ARGUS_VALIDATION_OWNER_TOKEN=$META_OWNER_TOKEN"; then
      # A member can disappear between stat and environ while the group is
      # shutting down. Recheck before treating it as an ownership violation.
      read_proc_identity "$pid" || continue
      [[ "$PROC_STATE" != "Z" ]] || continue
      echo "Process $pid joined the recorded group without its owner token; refusing signals" >&2
      return 1
    fi
    OWNED_GROUP_MEMBERS+=("$pid")
  done
}

go2rtc_process_matches() {
  local pid=$1
  local expected_exe actual_exe
  local -a args=()
  read_proc_identity "$pid" || return 1
  [[ "$PROC_STATE" != "Z" \
    && "$PROC_PGID" == "$META_PGID" \
    && "$PROC_SID" == "$META_SID" ]] || return 1
  process_environment_has "$pid" \
    "ARGUS_VALIDATION_OWNER_TOKEN=$META_OWNER_TOKEN" || return 1
  mapfile -d '' -t args < "/proc/$pid/cmdline"
  [[ ${#args[@]} -eq 3 \
    && "${args[1]:-}" == "-config" \
    && "${args[2]:-}" == "$GO2RTC_GENERATED_CONFIG" ]] || return 1
  expected_exe=$(realpath -- "$GO2RTC_BIN")
  actual_exe=$(realpath -- "/proc/$pid/exe") || return 1
  [[ "$actual_exe" == "$expected_exe" ]]
}

go2rtc_owned_pid() {
  collect_owned_group_members || return 1
  local pid found=""
  for pid in "${OWNED_GROUP_MEMBERS[@]}"; do
    go2rtc_process_matches "$pid" || continue
    [[ -z "$found" ]] || {
      echo "Multiple owned go2rtc processes found in one runtime group" >&2
      return 1
    }
    found=$pid
  done
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

process_is_descendant() {
  local child=$1
  local ancestor=$2
  local current=$child
  local depth=0
  while [[ "$current" =~ ^[0-9]+$ && "$current" -gt 1 && "$depth" -lt 128 ]]; do
    [[ "$current" == "$ancestor" ]] && return 0
    read_proc_identity "$current" || return 1
    current=$PROC_PPID
    ((depth += 1))
  done
  return 1
}

camera_owner_pids() {
  local proc pid fd target
  declare -A seen=()
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    [[ -d "$proc/fd" ]] || continue
    for fd in "$proc"/fd/*; do
      [[ -e "$fd" || -L "$fd" ]] || continue
      target=$(readlink -f -- "$fd" 2>/dev/null || true)
      [[ "$target" == "$CAMERA_DEVICE" ]] || continue
      seen[$pid]=1
      break
    done
  done
  for pid in "${!seen[@]}"; do
    printf '%s\n' "$pid"
  done
}

assert_camera_unowned() {
  local owners
  owners=$(camera_owner_pids)
  [[ -z "$owners" ]] || {
    echo "Camera is already open by unverified PID(s): ${owners//$'\n'/ }" >&2
    return 1
  }
}

verify_camera_owner() {
  local go2rtc_pid owners owner cmdline matched=0
  go2rtc_pid=$(go2rtc_owned_pid) || {
    echo "FAIL no exactly-owned go2rtc process" >&2
    return 1
  }
  owners=$(camera_owner_pids)
  [[ -n "$owners" ]] || {
    echo "FAIL camera has no capture owner" >&2
    return 1
  }
  while IFS= read -r owner; do
    [[ -n "$owner" ]] || continue
    process_is_descendant "$owner" "$go2rtc_pid" || {
      echo "FAIL camera owner $owner is not a descendant of go2rtc $go2rtc_pid" >&2
      return 1
    }
    read_proc_identity "$owner" || return 1
    [[ "$PROC_PGID" == "$META_PGID" && "$PROC_SID" == "$META_SID" ]] || {
      echo "FAIL camera owner $owner escaped the validation process group" >&2
      return 1
    }
    process_environment_has "$owner" \
      "ARGUS_VALIDATION_OWNER_TOKEN=$META_OWNER_TOKEN" || {
      echo "FAIL camera owner $owner lacks the validation owner token" >&2
      return 1
    }
    cmdline=$(tr '\0' ' ' < "/proc/$owner/cmdline")
    if [[ "$cmdline" == *"$CAMERA_DEVICE"* \
      && "$cmdline" == *"-input_format mjpeg"* \
      && "$cmdline" == *"-video_size 1920x1080"* \
      && "$cmdline" == *"-framerate 60"* \
      && "$cmdline" == *"-c:v h264_nvenc"* ]]; then
      matched=1
    fi
  done <<< "$owners"
  [[ "$matched" == 1 ]] || {
    echo "FAIL camera owner does not prove MJPEG 1080p60 + NVENC" >&2
    return 1
  }
}

wait_camera_owner() {
  # go2rtc's exec source is lazy: the FFmpeg/NVENC child is created only
  # after the pipeline opens the RTSP stream.  The Dashboard and go2rtc API
  # listeners can therefore be healthy before /dev/video0 has an owner.
  # Treat that interval as startup-in-progress, while keeping the exact
  # descendant/process-group/token/command-line checks fail-closed.
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until verify_camera_owner >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      verify_camera_owner || true
      return 1
    fi
    sleep 1
  done
}

terminate_owned_runtime() {
  local label=${1:-Argus}
  if [[ ! -e "$ARGUS_PROCESS_METADATA" ]]; then
    echo "$label: not running"
    return 0
  fi
  load_process_metadata || {
    echo "$label: invalid metadata; refusing to signal anything" >&2
    return 1
  }
  local pgid=$META_PGID
  local leader_exact=0
  process_matches_metadata && leader_exact=1
  if (( leader_exact == 0 )) \
    && read_proc_identity "$META_PID" \
    && [[ "$PROC_STATE" != "Z" && "$PROC_STARTTIME" == "$META_STARTTIME" ]]; then
    echo "$label: recorded leader is still alive but failed full identity verification; refusing signals" >&2
    return 1
  fi
  collect_owned_group_members || return 1
  if (( ${#OWNED_GROUP_MEMBERS[@]} == 0 )); then
    rm -f -- "$ARGUS_PROCESS_METADATA"
    echo "$label: stale metadata removed; no live group members"
    return 0
  fi
  if (( leader_exact == 0 )); then
    echo "$label: leader is gone; cleaning ${#OWNED_GROUP_MEMBERS[@]} exactly-owned crash orphan(s)"
  fi

  if ! kill -TERM -- "-$pgid" 2>/dev/null; then
    collect_owned_group_members || return 1
    (( ${#OWNED_GROUP_MEMBERS[@]} == 0 )) || {
      echo "$label: verified group could not receive TERM" >&2
      return 1
    }
  fi
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    collect_owned_group_members || return 1
    (( ${#OWNED_GROUP_MEMBERS[@]} > 0 )) || break
    sleep 1
  done
  collect_owned_group_members || return 1
  if (( ${#OWNED_GROUP_MEMBERS[@]} > 0 )); then
    echo "$label: graceful group stop timed out; killing verified process group $pgid" >&2
    if ! kill -KILL -- "-$pgid" 2>/dev/null; then
      collect_owned_group_members || return 1
      (( ${#OWNED_GROUP_MEMBERS[@]} == 0 )) || {
        echo "$label: verified group could not receive KILL" >&2
        return 1
      }
    fi
    deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
      collect_owned_group_members || return 1
      (( ${#OWNED_GROUP_MEMBERS[@]} > 0 )) || break
      sleep 1
    done
  fi
  collect_owned_group_members || return 1
  (( ${#OWNED_GROUP_MEMBERS[@]} == 0 )) || {
    echo "$label: exactly-owned process group still has live members" >&2
    return 1
  }
  rm -f -- "$ARGUS_PROCESS_METADATA"
  echo "$label: stopped"
}

on_signal() {
  local signal=$1
  trap - INT TERM HUP
  if (( STARTED_THIS_INVOCATION == 1 )); then
    echo "Interrupted by $signal; cleaning the runtime started by this invocation" >&2
    terminate_owned_runtime "Argus startup" || true
  fi
  case "$signal" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    HUP) exit 129 ;;
  esac
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

launch_runtime() {
  [[ ! -e "$LEGACY_PID_FILE" ]] || {
    echo "Legacy PID file exists without strong metadata: $LEGACY_PID_FILE" >&2
    return 1
  }
  local owner_token handshake deadline launched_pid
  owner_token=$(tr -d '-' < /proc/sys/kernel/random/uuid)
  owner_token+=$(tr -d '-' < /proc/sys/kernel/random/uuid)
  handshake="$PID_DIR/.argus-launch-${owner_token}"
  rm -f -- "$handshake" "$ARGUS_PROCESS_METADATA"

  env \
    ARGUS_PROCESS_METADATA="$ARGUS_PROCESS_METADATA" \
    ARGUS_LAUNCH_HANDSHAKE="$handshake" \
    ARGUS_VALIDATION_OWNER_TOKEN="$owner_token" \
    ARGUS_STATE_DIR="$ARGUS_STATE_DIR" \
    ARGUS_CONFIG="$ARGUS_CONFIG" \
    ARGUS_PYTHON="$ARGUS_PYTHON" \
    PYTHONPATH="$ARGUS_ROOT/src" \
    PYTHONPYCACHEPREFIX="$ARGUS_STATE_DIR/pycache" \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME="$ARGUS_STATE_DIR/cache/xdg" \
    TORCH_HOME="$ARGUS_STATE_DIR/cache/torch" \
    ULTRALYTICS_CONFIG_DIR="$ARGUS_STATE_DIR/cache/ultralytics" \
    MPLCONFIGDIR="$ARGUS_STATE_DIR/cache/matplotlib" \
    HF_HOME="$ARGUS_STATE_DIR/cache/huggingface" \
    ARGUS__DASHBOARD__HOST=127.0.0.1 \
    ARGUS__DASHBOARD__PORT="$ARGUS_PORT" \
    ARGUS__DASHBOARD__GO2RTC_ENABLED=true \
    ARGUS__DASHBOARD__GO2RTC_API_PORT="$GO2RTC_API_PORT" \
    ARGUS__DASHBOARD__GO2RTC_RTSP_PORT="$GO2RTC_RTSP_PORT" \
    ARGUS__DASHBOARD__GO2RTC_WEBRTC_PORT="$GO2RTC_WEBRTC_PORT" \
    ARGUS__DASHBOARD__GO2RTC_WEBRTC_LISTEN_HOST=127.0.0.1 \
    ARGUS__DASHBOARD__GO2RTC_BINARY="$GO2RTC_BIN" \
    ARGUS__DASHBOARD__GO2RTC_CONFIG_DIR="$GO2RTC_CONFIG_DIR" \
    ARGUS__DASHBOARD__GO2RTC_REPLACE_STALE_PROCESS=false \
    setsid --fork bash -c '
      set -Eeuo pipefail
      umask 077
      pid=$$
      IFS= read -r stat_line < "/proc/$pid/stat"
      rest=${stat_line##*) }
      read -r -a fields <<< "$rest"
      (( ${#fields[@]} >= 20 ))
      pgid=${fields[2]}
      sid=${fields[3]}
      starttime=${fields[19]}
      [[ "$pid" == "$pgid" && "$pid" == "$sid" ]]
      boot_id=$(< /proc/sys/kernel/random/boot_id)
      tmp="${ARGUS_PROCESS_METADATA}.tmp.${pid}"
      {
        printf "version=1\n"
        printf "boot_id=%s\n" "$boot_id"
        printf "pid=%s\n" "$pid"
        printf "starttime=%s\n" "$starttime"
        printf "pgid=%s\n" "$pgid"
        printf "sid=%s\n" "$sid"
        printf "owner_token=%s\n" "$ARGUS_VALIDATION_OWNER_TOKEN"
      } > "$tmp"
      chmod 600 "$tmp"
      mv -f -- "$tmp" "$ARGUS_PROCESS_METADATA"
      printf "%s\n" "$pid" > "$ARGUS_LAUNCH_HANDSHAKE"
      cd "$ARGUS_STATE_DIR"
      exec "$ARGUS_PYTHON" -m argus --config "$ARGUS_CONFIG"
    ' </dev/null >>"$ARGUS_LOG" 2>&1 9>&-

  deadline=$((SECONDS + 10))
  until [[ -s "$handshake" && -s "$ARGUS_PROCESS_METADATA" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Independent Argus session did not publish launch metadata" >&2
      rm -f -- "$handshake"
      terminate_owned_runtime "Argus startup" || true
      return 1
    fi
    sleep 0.1
  done
  launched_pid=$(< "$handshake")
  rm -f -- "$handshake"
  STARTED_THIS_INVOCATION=1
  load_process_metadata || {
    terminate_owned_runtime "Argus startup" || true
    return 1
  }
  [[ "$launched_pid" == "$META_PID" ]] || {
    echo "Launch handshake PID does not match process metadata" >&2
    terminate_owned_runtime "Argus startup" || true
    return 1
  }
  deadline=$((SECONDS + 10))
  until process_matches_metadata; do
    if (( SECONDS >= deadline )); then
      echo "Argus leader failed exact post-exec identity verification" >&2
      terminate_owned_runtime "Argus startup" || true
      return 1
    fi
    sleep 0.1
  done
}

startup_failure() {
  local message=$1
  echo "$message" >&2
  diagnose_runtime || true
  if (( STARTED_THIS_INVOCATION == 1 )); then
    terminate_owned_runtime "Argus startup" || true
    STARTED_THIS_INVOCATION=0
  fi
  return 1
}

start_runtime() {
  verify_artifacts || { echo "Runtime artifact SHA256 verification failed" >&2; return 1; }
  verify_source || { echo "Strict source inventory/SHA256 verification failed" >&2; return 1; }
  [[ -e "$CAMERA_DEVICE" ]] || { echo "Camera missing: $CAMERA_DEVICE" >&2; return 1; }
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fqx "NVIDIA GeForce RTX 3060" || {
    echo "The required RTX 3060 is not visible" >&2
    return 1
  }

  if process_matches_metadata; then
    echo "Argus: already running with exact validation identity"
  else
    if [[ -e "$ARGUS_PROCESS_METADATA" ]]; then
      echo "Found stopped leader metadata; checking for exactly-owned crash orphans"
      terminate_owned_runtime "Argus crash recovery" || return 1
    fi
    [[ ! -e "$LEGACY_PID_FILE" ]] || {
      echo "Legacy PID file prevents safe startup: $LEGACY_PID_FILE" >&2
      return 1
    }
    assert_camera_unowned || return 1
    assert_ports_free || return 1
    launch_runtime || return 1
  fi

  wait_api "/api/system/health" || {
    startup_failure "Argus did not become healthy"
    return 1
  }
  wait_http "http://127.0.0.1:${GO2RTC_API_PORT}/api/streams" || {
    startup_failure "Argus-managed go2rtc did not become healthy"
    return 1
  }
  go2rtc_owned_pid >/dev/null || {
    startup_failure "go2rtc API is reachable but has no exact owned process"
    return 1
  }
  wait_camera_owner || {
    startup_failure "Camera ownership did not converge"
    return 1
  }
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  local health_output=""
  until health_output=$(health_runtime 2>&1); do
    if (( SECONDS >= deadline )); then
      echo "$health_output" >&2
      echo "SAFETY-BLOCKED: infrastructure is online, but the complete " \
        "SceneGuard/model/identity gate did not pass; keeping the isolated " \
        "runtime online for authenticated preview and diagnosis" >&2
      STARTED_THIS_INVOCATION=0
      diagnose_runtime || true
      return 1
    fi
    sleep 2
  done
  STARTED_THIS_INVOCATION=0
  echo "$health_output"
}

stop_runtime() {
  terminate_owned_runtime "Argus"
  STARTED_THIS_INVOCATION=0
}

print_process_status() {
  if process_matches_metadata; then
    echo "Argus: running pid=$META_PID pgid=$META_PGID sid=$META_SID starttime=$META_STARTTIME boot_id=$META_BOOT_ID"
    return
  fi
  if [[ -e "$ARGUS_PROCESS_METADATA" ]]; then
    if collect_owned_group_members 2>/dev/null && (( ${#OWNED_GROUP_MEMBERS[@]} > 0 )); then
      echo "Argus: leader mismatch with ${#OWNED_GROUP_MEMBERS[@]} exactly-owned crash orphan(s)"
    else
      echo "Argus: stopped, invalid metadata, or ownership mismatch"
    fi
  else
    echo "Argus: stopped"
  fi
}

status_runtime() {
  echo "source_snapshot=$ARGUS_ROOT"
  echo "state_dir=$ARGUS_STATE_DIR"
  echo "source_sha256=$(verify_source >/dev/null 2>&1 && echo ok || echo mismatch)"
  echo "artifact_sha256=$(verify_artifacts >/dev/null 2>&1 && echo ok || echo mismatch)"
  print_process_status
  local go2rtc_pid=""
  go2rtc_pid=$(go2rtc_owned_pid 2>/dev/null || true)
  if [[ -n "$go2rtc_pid" ]]; then
    echo "go2rtc: running as exactly-owned group member pid=$go2rtc_pid"
  else
    echo "go2rtc: stopped or ownership mismatch"
  fi
  if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --device "$CAMERA_DEVICE" --get-fmt-video 2>&1 || true
    v4l2-ctl --device "$CAMERA_DEVICE" --get-parm 2>&1 || true
  else
    echo "v4l2-ctl: unavailable"
  fi
  nvidia-smi \
    --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>&1 || true
  echo "tcp_listeners:"
  ss -H -ltn 2>/dev/null \
    | grep -E ":(${ARGUS_PORT}|${GO2RTC_API_PORT}|${GO2RTC_RTSP_PORT}|${GO2RTC_WEBRTC_PORT})\\b" \
    || true
  echo "udp_listeners:"
  ss -H -lun 2>/dev/null \
    | grep -E ":(${ARGUS_PORT}|${GO2RTC_API_PORT}|${GO2RTC_RTSP_PORT}|${GO2RTC_WEBRTC_PORT})\\b" \
    || true
  echo "go2rtc_streams:"
  curl --silent --show-error --max-time 5 \
    "http://127.0.0.1:${GO2RTC_API_PORT}/api/streams" 2>&1 || true
  echo
  echo "argus_health:"
  curl --silent --show-error --max-time 5 \
    "http://127.0.0.1:${ARGUS_PORT}/api/system/health" 2>&1 || true
  echo
}

health_runtime() {
  verify_artifacts || { echo "FAIL runtime artifact SHA256 mismatch" >&2; return 1; }
  verify_source || { echo "FAIL strict source inventory/SHA256 mismatch" >&2; return 1; }
  process_matches_metadata || {
    echo "FAIL Argus process identity/ownership" >&2
    return 1
  }
  go2rtc_owned_pid >/dev/null || {
    echo "FAIL go2rtc is not an exactly-owned process-group member" >&2
    return 1
  }
  verify_camera_owner || return 1

  assert_loopback_listener tcp "$ARGUS_PORT" || return 1
  assert_loopback_listener tcp "$GO2RTC_API_PORT" || return 1
  assert_loopback_listener tcp "$GO2RTC_RTSP_PORT" || return 1
  assert_loopback_listener tcp "$GO2RTC_WEBRTC_PORT" || return 1
  assert_loopback_listener udp "$GO2RTC_WEBRTC_PORT" || return 1

  local health models registry detail streaming degradation streams
  local unauth_file unauth_status unauth
  health=$(curl --silent --show-error --fail --max-time 10 \
    "http://127.0.0.1:${ARGUS_PORT}/api/system/health")
  models=$(curl_api "/api/models/status")
  registry=$(curl_api "/api/models/json")
  detail=$(curl_api "/api/cameras/${CAMERA_ID}/detail/json")
  streaming=$(curl_api "/api/streaming/${CAMERA_ID}")
  degradation=$(curl_api "/api/system/anomaly-degradation")
  streams=$(curl --silent --show-error --fail --max-time 10 \
    "http://127.0.0.1:${GO2RTC_API_PORT}/api/streams")

  unauth_file="$PID_DIR/.unauthorized-${BASHPID}.json"
  rm -f -- "$unauth_file"
  unauth_status=$(curl --silent --show-error --max-time 10 \
    --output "$unauth_file" --write-out '%{http_code}' \
    "http://127.0.0.1:${ARGUS_PORT}/api/models/json") || {
      rm -f -- "$unauth_file"
      echo "FAIL unauthenticated API probe could not complete" >&2
      return 1
    }
  unauth=$(< "$unauth_file")
  rm -f -- "$unauth_file"
  [[ "$unauth_status" == "401" ]] || {
    echo "FAIL protected API without a token returned HTTP $unauth_status, expected 401" >&2
    return 1
  }

  HEALTH_JSON=$health MODELS_JSON=$models REGISTRY_JSON=$registry \
    DETAIL_JSON=$detail STREAMING_JSON=$streaming DEGRADATION_JSON=$degradation \
    GO2RTC_STREAMS_JSON=$streams UNAUTH_JSON=$unauth \
    CAMERA_ID_VALUE=$CAMERA_ID CAMERA_DEVICE_VALUE=$CAMERA_DEVICE \
    ARGUS_PORT_VALUE=$ARGUS_PORT GO2RTC_API_PORT_VALUE=$GO2RTC_API_PORT \
    GO2RTC_RTSP_PORT_VALUE=$GO2RTC_RTSP_PORT \
    GO2RTC_WEBRTC_PORT_VALUE=$GO2RTC_WEBRTC_PORT \
    GO2RTC_CONFIG_VALUE=$GO2RTC_GENERATED_CONFIG \
    ARGUS_STATE_DIR_VALUE=$ARGUS_STATE_DIR \
    MODEL_VERSION_ID_VALUE=$MODEL_VERSION_ID MODEL_STAGE_VALUE=$MODEL_STAGE \
    RUNTIME_ARTIFACT_SHA256_VALUE=$RUNTIME_ARTIFACT_SHA256 \
    SCENE_PROFILE_ID_VALUE=$SCENE_PROFILE_ID \
    RELEASE_REVISION_VALUE=$RELEASE_REVISION \
    "$ARGUS_PYTHON" -I -S - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


def envelope(name: str):
    try:
        body = json.loads(os.environ[name])
    except (KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAIL {name}: invalid JSON: {exc}") from exc
    if (
        not isinstance(body, dict)
        or set(body) != {"code", "msg", "data"}
        or body.get("code") != 0
        or not isinstance(body.get("msg"), str)
    ):
        raise SystemExit(f"FAIL {name}: invalid success envelope: {body!r}")
    if not isinstance(body.get("data"), dict):
        raise SystemExit(f"FAIL {name}: data must be an object")
    return body["data"]


camera_id = os.environ["CAMERA_ID_VALUE"]
camera_device = os.environ["CAMERA_DEVICE_VALUE"]
argus_port = int(os.environ["ARGUS_PORT_VALUE"])
api_port = int(os.environ["GO2RTC_API_PORT_VALUE"])
rtsp_port = int(os.environ["GO2RTC_RTSP_PORT_VALUE"])
webrtc_port = int(os.environ["GO2RTC_WEBRTC_PORT_VALUE"])
state_dir = Path(os.environ["ARGUS_STATE_DIR_VALUE"]).resolve(strict=True)
model_version = os.environ["MODEL_VERSION_ID_VALUE"]
model_stage = os.environ["MODEL_STAGE_VALUE"]
artifact_sha256 = os.environ["RUNTIME_ARTIFACT_SHA256_VALUE"].lower()
profile_id = os.environ["SCENE_PROFILE_ID_VALUE"]
release_revision = int(os.environ["RELEASE_REVISION_VALUE"])

unauth = json.loads(os.environ["UNAUTH_JSON"])
if (
    not isinstance(unauth, dict)
    or set(unauth) != {"code", "msg", "data"}
    or unauth.get("code") != 40100
    or not isinstance(unauth.get("msg"), str)
    or not isinstance(unauth.get("data"), dict)
    or not unauth["data"].get("request_id")
):
    raise SystemExit(f"FAIL unauthenticated 401 envelope: {unauth!r}")

health = envelope("HEALTH_JSON")
if health.get("status") != "healthy":
    raise SystemExit(f"FAIL Argus health={health.get('status')!r}")
camera_health = next(
    (item for item in health.get("cameras", []) if item.get("camera_id") == camera_id),
    None,
)
if not camera_health or camera_health.get("connected") is not True:
    raise SystemExit(f"FAIL camera health is not connected: {camera_health!r}")

registry = envelope("REGISTRY_JSON")
records = registry.get("models")
if not isinstance(records, list):
    raise SystemExit("FAIL registry model list is missing")
if any(item.get("stage") == "production" for item in records if isinstance(item, dict)):
    raise SystemExit("FAIL isolated registry contains a production-stage model")
camera_records = [
    item for item in records
    if isinstance(item, dict) and item.get("camera_id") == camera_id
]
matches = [item for item in camera_records if item.get("model_version_id") == model_version]
if len(matches) != 1 or len(camera_records) != 1:
    raise SystemExit(
        f"FAIL expected one exact isolated registry record, got {camera_records!r}"
    )
record = matches[0]
if (
    record.get("stage") != model_stage
    or record.get("stage") != "canary"
    or record.get("canary_camera_id") != camera_id
    or record.get("is_active") is not False
    or record.get("runtime_state") != "applied"
    or record.get("release_revision") != release_revision
):
    raise SystemExit(f"FAIL registry release identity mismatch: {record!r}")
try:
    training_params = json.loads(record.get("training_params") or "null")
except json.JSONDecodeError as exc:
    raise SystemExit("FAIL registry training_params is invalid JSON") from exc
if (
    not isinstance(training_params, dict)
    or training_params.get("authority") != "isolated-validation-only"
    or training_params.get("production_approved") is not False
):
    raise SystemExit(f"FAIL registry validation authority mismatch: {training_params!r}")
model_path = Path(str(record.get("model_path") or "")).resolve(strict=True)
try:
    model_path.relative_to(state_dir)
except ValueError as exc:
    raise SystemExit(f"FAIL registry model escaped validation state: {model_path}") from exc

models = envelope("MODELS_JSON").get("models", [])
anomaly = next(
    (
        item for item in models
        if item.get("camera_id") == camera_id
        and str(item.get("name", "")).lower().startswith("anomaly")
    ),
    None,
)
if not anomaly:
    raise SystemExit("FAIL anomaly model status is missing")
if (
    anomaly.get("backend") != "torch-cuda"
    or anomaly.get("loaded") is not True
    or Path(str(anomaly.get("model_path") or "")).resolve(strict=True) != model_path
):
    raise SystemExit(f"FAIL CUDA model/path is not applied: {anomaly!r}")

detail = envelope("DETAIL_JSON")
expected_runtime_source = f"rtsp://127.0.0.1:{rtsp_port}/{camera_id}"
if (
    detail.get("camera_id") != camera_id
    or detail.get("running") is not True
    or detail.get("connected") is not True
    or detail.get("declared_source") != camera_device
    or detail.get("runtime_source") != expected_runtime_source
    or detail.get("go2rtc_stream") != camera_id
    or detail.get("source_consistency") != "go2rtc_redirect"
    or detail.get("model_backend") != "torch-cuda"
    or detail.get("expected_model_backend") != "torch-cuda"
):
    raise SystemExit(f"FAIL camera runtime/source identity mismatch: {detail!r}")
scene_guard = detail.get("scene_guard")
if (
    not isinstance(scene_guard, dict)
    or scene_guard.get("usable") is not True
    or scene_guard.get("profile_id") != profile_id
    or scene_guard.get("bound_model_version") != model_version
    or str(scene_guard.get("bound_model_artifact_sha256", "")).lower()
       != artifact_sha256
    or str(scene_guard.get("runtime_model_artifact_sha256", "")).lower()
       != artifact_sha256
    or scene_guard.get("reference_available") is not True
):
    raise SystemExit(f"FAIL SceneGuard identity is not usable/exact: {scene_guard!r}")
runner = detail.get("runner")
if (
    not isinstance(runner, dict)
    or runner.get("version_tag") != model_version
    or Path(str(runner.get("model_ref") or "")).resolve(strict=True) != model_path
):
    raise SystemExit(f"FAIL live runner identity mismatch: {runner!r}")

degradation = envelope("DEGRADATION_JSON").get("anomaly")
if not isinstance(degradation, dict) or degradation.get("degraded") is not False:
    raise SystemExit(f"FAIL anomaly runtime is degraded: {degradation!r}")
camera_degradation = next(
    (
        item for item in degradation.get("cameras", [])
        if item.get("camera_id") == camera_id
    ),
    None,
)
if camera_degradation is not None and camera_degradation.get("degraded") is not False:
    raise SystemExit(f"FAIL camera anomaly runtime is degraded: {camera_degradation!r}")

streaming = envelope("STREAMING_JSON")
if streaming.get("go2rtc") is not True:
    raise SystemExit(f"FAIL go2rtc browser stream unavailable: {streaming!r}")
expected_ws_prefix = f"ws://127.0.0.1:{argus_port}/ws/stream/{camera_id}"
for key, mode in (("webrtc_ws", "webrtc"), ("mse_ws", "mse")):
    wanted = f"{expected_ws_prefix}?mode={mode}"
    if streaming.get(key) != wanted:
        raise SystemExit(f"FAIL {key} is not exact same-origin URL: {streaming.get(key)!r}")

try:
    go2rtc_streams = json.loads(os.environ["GO2RTC_STREAMS_JSON"])
except json.JSONDecodeError as exc:
    raise SystemExit("FAIL go2rtc /api/streams did not return JSON") from exc
if not isinstance(go2rtc_streams, dict) or camera_id not in go2rtc_streams:
    raise SystemExit(f"FAIL go2rtc stream is not registered: {go2rtc_streams!r}")

expected_exec = (
    "exec:ffmpeg -hide_banner -loglevel warning -f v4l2 "
    "-input_format mjpeg -video_size 1920x1080 -framerate 60 "
    f"-i {camera_device} -an -c:v h264_nvenc -preset ll -tune ull "
    "-rc constqp -qp 24 -pix_fmt yuv420p -f rtsp "
    "-rtsp_transport tcp {output}"
    "#killsignal=sigkill#video=h264#starttimeout=20"
)
go2rtc_config_path = Path(os.environ["GO2RTC_CONFIG_VALUE"]).resolve(strict=True)
try:
    go2rtc_config_path.relative_to(state_dir)
except ValueError as exc:
    raise SystemExit("FAIL generated go2rtc config escaped validation state") from exc
go2rtc_config = json.loads(go2rtc_config_path.read_text(encoding="utf-8"))
expected_config = {
    "api": {
        "listen": f"127.0.0.1:{api_port}",
        "origin": "http://127.0.0.1",
    },
    "rtsp": {"listen": f"127.0.0.1:{rtsp_port}"},
    "webrtc": {"listen": f"127.0.0.1:{webrtc_port}"},
    "streams": {camera_id: [expected_exec]},
}
if go2rtc_config != expected_config:
    raise SystemExit(
        "FAIL generated go2rtc config is not the exact loopback/NVENC contract: "
        f"{go2rtc_config!r}"
    )

profile_root = state_dir / "scene_profiles" / camera_id
active = json.loads((profile_root / "active.json").read_text(encoding="utf-8"))
profile_document = json.loads(
    (profile_root / profile_id / "metadata.json").read_text(encoding="utf-8")
)
if active.get("profile_id") != profile_id or active.get("camera_id") != camera_id:
    raise SystemExit(f"FAIL active SceneGuard pointer mismatch: {active!r}")
metadata = profile_document.get("metadata")
if (
    profile_document.get("profile_id") != profile_id
    or profile_document.get("camera_id") != camera_id
    or not isinstance(metadata, dict)
    or metadata.get("validation_only") is not True
    or metadata.get("operator_approved") is not False
    or metadata.get("model_version_id") != model_version
    or str(metadata.get("model_artifact_sha256", "")).lower() != artifact_sha256
    or metadata.get("authority") != "isolated-validation-only"
):
    raise SystemExit(f"FAIL stored validation SceneGuard authority mismatch: {metadata!r}")

print(
    "PASS strict source/artifact identity, exact process group, loopback TCP/UDP, "
    "OBSBOT NVENC stream, authenticated API, canary registry/runner, SceneGuard, "
    "and torch-cuda model are healthy"
)
PY
}

diagnose_runtime() {
  echo "===== process status ====="
  print_process_status
  local go2rtc_pid=""
  go2rtc_pid=$(go2rtc_owned_pid 2>/dev/null || true)
  echo "go2rtc owned pid=${go2rtc_pid:-missing}"
  echo "===== generated go2rtc config ====="
  sed -n '1,200p' "$GO2RTC_GENERATED_CONFIG" 2>/dev/null || true
  echo "===== TCP listeners ====="
  ss -H -ltn 2>/dev/null \
    | grep -E ":(${ARGUS_PORT}|${GO2RTC_API_PORT}|${GO2RTC_RTSP_PORT}|${GO2RTC_WEBRTC_PORT})\\b" \
    || true
  echo "===== UDP listeners ====="
  ss -H -lun 2>/dev/null \
    | grep -E ":(${ARGUS_PORT}|${GO2RTC_API_PORT}|${GO2RTC_RTSP_PORT}|${GO2RTC_WEBRTC_PORT})\\b" \
    || true
  echo "===== recent Argus log ====="
  tail -n 160 "$ARGUS_LOG" 2>/dev/null || true
  echo "===== failure signatures ====="
  grep -Eini \
    'rtsp|decode|invalid data|buffer|overflow|queue full|cuda|traceback|fatal|killed|out of memory|oom' \
    "$ARGUS_LOG" 2>/dev/null | tail -n 200 || true
}

case "$COMMAND" in
  start) start_runtime ;;
  stop) stop_runtime ;;
  restart) stop_runtime; start_runtime ;;
  status) status_runtime ;;
  health) health_runtime ;;
  diagnose) diagnose_runtime ;;
  *) usage; exit 2 ;;
esac
