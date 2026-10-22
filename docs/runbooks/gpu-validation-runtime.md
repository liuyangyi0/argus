# gpu-validation 测试机运行手册

最后更新：2026-07-14

## 示例环境（不是可连接的部署清单）

本页的主机、账号和路径均为公开示例。真实部署清单、SSH 配置、凭据与
模型位置应保存在仓库外。`192.0.2.0/24` 和 `.example` 仅用于文档。
脚本默认禁止写入运行账号的 `~/argus`；部署在其他路径时还必须通过
`--forbidden-root` 明确保护实际生产目录，不能将示例地址当成真实配置。

- 主机：`gpu-validation` / `192.0.2.10`
- 历史主仓库（隔离验证禁止写入）：`/srv/argus`
- 隔离源码快照：`/srv/argus-validation/<run>/source`
- 隔离运行状态：`/srv/argus-validation/<run>/state`
- 回归素材源：`/srv/argus-regression-capture`
- 摄像头：OBSBOT Meet 2，Linux 设备 `/dev/video0`
- 采集档位：MJPEG `1920x1080` `60 FPS`
- GPU：NVIDIA RTX 3060
- Linux go2rtc 原始 binary：`/srv/argus/data/runtime/go2rtc/v1.9.14/go2rtc`；
  每次验证先复制到独立 state 并记录 SHA256，不能使用仓库里的 Windows
  `bin/go2rtc.exe`

## 运行原则

- OpenVINO `model.xml` 不是 NVIDIA CUDA 主模型；它只能证明 OpenVINO 后端可用。
- 有 NVIDIA GPU 且目标是 GPU 主模型时，配置必须使用 `anomaly.inference_backend: torch-cuda`，并发布 `model.pt`。
- `torch-cuda` 不能静默回退 CPU、OpenVINO 或 SSIM fallback；发现 artifact 不匹配或 CUDA 不可用时必须在 Web 上显示红色降级。
- gpu-validation 当前默认运行策略：强制 `torch-cuda`。即使 Torch/CUDA 回归数字暂时差，也不能因为测试机有 GPU 就偷偷跑 OpenVINO/CPU；质量问题必须通过回归报表暴露和修复。
- OpenVINO v002 只保留为 rollback/reference，不作为测试机当前默认运行模型。

## go2rtc 实时预览

隔离验证采用 Argus 管理的 go2rtc 子进程，但摄像头 producer 仍必须是
`exec:ffmpeg` + `h264_nvenc`。测试机 Ubuntu ffmpeg 4.4.2 不兼容 go2rtc
内置 `ffmpeg:device`。`usb.go2rtc_source` 会作为 initial-config stream 写入隔离
`go2rtc.json`；Argus 启动失败或 RTSP listener 未就绪时必须回收该子进程，并拒绝
直接打开 `/dev/video0`。

关键端口：

- go2rtc API / MSE：`11984`
- RTSP relay：`18554`
- WebRTC：`18555`
- Argus API：`18081`

检查：

```bash
scripts/gpu_validation_runtime.sh status --env <state>/validation.env
scripts/gpu_validation_runtime.sh health --env <state>/validation.env
```

## 隔离环境准备与启停

准备脚本会生成独立配置、数据库、安全 admin/viewer 凭据、复制后的 go2rtc 和
Torch artifact、逐文件 SHA256 清单，并通过唯一 `ModelReleaseService` 创建
validation-only SceneGuard profile 和 canary。它绝不创建 production stage，也不写
历史主仓库：

```bash
RUN=/srv/argus-validation/<run>
SOURCE="$RUN/source"
STATE="$RUN/state"
mkdir -p "$STATE" "$STATE/prep-pycache" "$STATE/cache"/{xdg,torch,ultralytics,matplotlib,huggingface}
chmod 0700 "$STATE"
cd "$SOURCE"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "$STATE/source.sha256"
chmod 0600 "$STATE/source.sha256"
chmod -R a-w "$SOURCE"

PYTHONPYCACHEPREFIX="$STATE/prep-pycache" \
PYTHONPATH="$SOURCE/src:$SOURCE" \
XDG_CACHE_HOME="$STATE/cache/xdg" \
TORCH_HOME="$STATE/cache/torch" \
ULTRALYTICS_CONFIG_DIR="$STATE/cache/ultralytics" \
MPLCONFIGDIR="$STATE/cache/matplotlib" \
HF_HOME="$STATE/cache/huggingface" \
/srv/argus/.venv/bin/python scripts/prepare_gpu_validation.py \
  --base-config "$SOURCE/configs/default.yaml" \
  --source-root "$PWD" \
  --forbidden-root /srv/argus \
  --source-manifest "$STATE/source.sha256" \
  --state-dir "$STATE" \
  --python /srv/argus/.venv/bin/python \
  --go2rtc-bin /srv/argus/data/runtime/go2rtc/v1.9.14/go2rtc \
  --model-artifact /srv/argus/data/exports/phase2_patchcore_v002_torch_cuda_20260613_170400/validation_camera_01/default/weights/torch/model.pt \
  --baseline-reference /srv/argus/data/baselines/validation_camera_01/default/v002_existing_subset450/<approved-reference>.png
```

准备阶段也会实际加载复制后的 `model.pt`，并校验其全部 sidecar 的 canonical
runtime SHA，因此上述 cache/pycache 环境变量不能省略；否则不可变源码快照可能被
Python 写入 `__pycache__`，或第三方库把缓存写到非隔离目录。`source.sha256` 是严格
文件集清单，源码中新增或缺失任何文件都会使启动失败。

`--baseline-reference` 不能直接使用 OBSBOT 打开后的第一帧：设备冷启动时曾返回
可解码但几乎全绿的损坏 MJPEG 帧。若需从当前现场采集 validation reference，至少
预热 120 帧后再保存，并先核对亮度、对比度和图像内容。准备脚本会用与运行时相同的
SceneGuard 参数让参考图连续自检到 `STABLE`；低纹理、遮挡、严重欠曝或损坏图会在
创建 registry/profile 之前失败。失败时必须改善现场或选择已有且可追溯的批准基线，
禁止降低 SceneGuard 特征和质量门槛。

启动、停止、重启与诊断只能通过 PID ownership 脚本；禁止 `pkill -f`、`killall`
或按端口杀进程：

```bash
scripts/gpu_validation_runtime.sh start --env <state>/validation.env
scripts/gpu_validation_runtime.sh status --env <state>/validation.env
scripts/gpu_validation_runtime.sh diagnose --env <state>/validation.env
scripts/gpu_validation_runtime.sh restart --env <state>/validation.env
scripts/gpu_validation_runtime.sh stop --env <state>/validation.env
```

go2rtc 的 `exec:ffmpeg` producer 按需启动，Dashboard/API listener 可能先于
`/dev/video0` owner 就绪。生命周期脚本会在 `STARTUP_TIMEOUT_SECONDS` 内等待 producer，
但每次轮询仍严格核对它是 go2rtc 后代、属于同一隔离进程组、携带 owner token，且命令
行保持 MJPEG 1080p60 + NVENC；超时仍按失败处理，不能用一次性的 `sleep` 或跳过 owner
校验来掩盖冷启动竞态。

`validation.env` 和独立 `api-token` 权限均为 `0600`；前者包含 Playwright
admin/viewer 凭据，后者供 soak collector 读取。二者都不得打印、提交或复制到源码目录。
检查 API 时先安全加载该文件：

```bash
set -a
source <state>/validation.env
set +a
curl -s -H "X-API-Token: $ARGUS_API_TOKEN" http://127.0.0.1:18081/api/system/health
curl -s -H "X-API-Token: $ARGUS_API_TOKEN" http://127.0.0.1:18081/api/models/status
curl -s -H "X-API-Token: $ARGUS_API_TOKEN" http://127.0.0.1:18081/api/streaming/validation_camera_01
```

## 本地隧道

只在需要本机浏览器访问测试机时开启：

```powershell
ssh -N -L 18081:127.0.0.1:18081 operator@gpu-host.example
```

关闭本地残留隧道：

```powershell
Get-Process ssh
Stop-Process -Id <pid>
```

不要转发 go2rtc 的 `11984`；浏览器只能通过 Argus 同源 `/ws/stream/...`
鉴权网关访问。临时隧道是否存在不能作为服务状态依据。

## 历史正式实例模型链接（隔离验证阶段禁止执行）

以下命令只记录旧实例现状，不属于本轮验证步骤。baseline v3、完整回归和可信
72 小时 soak 全部通过前 production 为硬性 NO-GO，禁止执行链接切换或重启
`/srv/argus` 下的正式实例。

Torch/CUDA 当前默认：

```bash
cd /srv/argus
ln -sfn \
  /srv/argus/data/exports/phase2_patchcore_v002_torch_cuda_20260613_170400/validation_camera_01 \
  /srv/argus/data/exports/validation_camera_01
```

runtime config 必须显式包含：

```yaml
cameras:
  - camera_id: validation_camera_01
    anomaly:
      inference_backend: torch-cuda
```

`anomaly` 是摄像头级配置；写在 YAML 顶层不会覆盖该摄像头的推理后端。

已知 2026-06-13 快速样本回归未达标，这是需要继续修的质量问题，不是切回 OpenVINO 的默认理由：

- `FP/hour 5.5879`
- `recall 0.2`
- `P95 delay 7.7s`

OpenVINO v002 rollback/reference：

```bash
cd /srv/argus
ln -sfn \
  /srv/argus/data/exports/phase2_patchcore_v002_subset_20260613_153551/validation_camera_01 \
  /srv/argus/data/exports/validation_camera_01
```

OpenVINO v002 对应回归数字：

- `FP/hour 2.7256`
- `recall 1.0`
- `P95 delay 4.55s`

## 回归验证

OpenVINO v002 仅保留为 detector 基线/reference。必须显式传模型，避免把 SSIM fallback 的结果误记成 OpenVINO：

```bash
cd /srv/argus
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python scripts/run_regression_suite.py \
  --manifest data/regression/manifest.yaml \
  --evaluation-mode detector \
  --model /srv/argus/data/exports/phase2_patchcore_v002_subset_20260613_153551/validation_camera_01/default/weights/openvino/model.xml \
  --sample-every 1 \
  --output-dir data/regression/reports/phase2-v002-online-check
```

Torch/CUDA 重新校准后必须使用完整 pipeline 复跑。配置中的摄像头、active scene profile、baseline 和 `--model-version-id` 必须严格绑定；active profile 还必须经过人工批准才能满足发布门：

```bash
cd /srv/argus
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python scripts/run_regression_suite.py \
  --manifest data/regression/manifest.yaml \
  --evaluation-mode pipeline \
  --config data/runtime/<torch-cuda-version>.yaml \
  --model /srv/argus/data/exports/<torch-cuda-version>/validation_camera_01/default/weights/torch/model.pt \
  --model-version-id <与-active-scene-profile-完全相同的版本> \
  --scene-profile-artifact /srv/argus/data/scene_profiles/validation_camera_01/<approved-profile-id> \
  --baseline-artifact /srv/argus/data/baselines/<release-baseline-v3> \
  --corpus-artifact /srv/argus-regression-capture \
  --sample-every 1 \
  --fail-on-acceptance \
  --output-dir data/regression/reports/<torch-cuda-version>-release
```

回归报告不是 production 授权本身。promotion API 不接受客户端拼装的
`release_evidence`（即使结构完整也拒绝）；production 请求必须提交
`release_evidence_sources`，分别指向测试机上的 regression report、实际 runtime
config、active scene profile 目录、训练 baseline 目录、完整 regression corpus 根目录、
soak report 和原始 soak telemetry。训练 baseline 必须是
`build_scene_profile_baseline.py` 生成的
`argus.scene-profile-baseline.v3` 目录（根含 `manifest.json`），并且必须由
`build_scene_profile_baseline.py --strict-release-decoder` 生成完整 FFprobe/FFmpeg
PTS 与 clean-EOF 证据；默认 v2 只可用于离线训练。corpus 根必须包含
`manifest.yaml` 以及清单引用的全部视频和标注。服务端使用
`argus.storage.release_evidence.build_release_evidence_from_sources()` 读取这些产物，
用 `evidence_artifact_sha256()` 重算 config/profile/baseline/corpus（目录会绑定排序
后的相对路径、文件大小和全部字节），并用 `runtime_artifact_sha256()` 重算实际
模型及 sidecar。regression report 的 `suite.evaluated_artifacts` 必须逐项匹配这些
服务端摘要且只能包含 registry 对应的单一 camera；逐案例
`evaluated_source_inventory` 和 `evaluated_scene_profile` 必须与 corpus 文件、当前
active profile 的 camera/profile/reference/baseline/model lineage 完全一致。服务端
还会把 baseline 整树重算为 registry `data_hash`，复核 manifest 内 reference/sample
摘要，并拒绝训练 source SHA 与 regression source SHA 的任何交集。soak report 也必须绑定同一
registry row、runtime、评估产物和 regression report。证据必须同时包含并通过：

- model version、camera、registry model hash，以及实际 `model_path` 重新计算出的完整 runtime artifact SHA256；
- config、active scene profile、baseline、完整 corpus、regression report 的 SHA256；
- 报告状态 `ok`、完整 pipeline 模式、可信 scene reference、runtime model identity、production parity、正样本标注覆盖与 candidate confirmation；
- 完整且独立的 source inventory（每段素材 FPS/帧数可知、读到 EOF、未被
  `--max-frames` 截断、无读取错误），每段有唯一 `source_session_id` 且不与训练/
  calibration session 重叠；至少 2 小时负样本、20 个 static-foreign 事件和 20 个
  projectile 事件；
- SceneGuard coverage ≥ 0.99、FP ≤ 2/h、overall event recall ≥ 0.95、static-foreign recall ≥ 0.95、P95 delay ≤ 10s；
- active scene profile 的独立人工批准，以及当前已认证发布操作者的独立 literal
  `operator_approved: true`（HTTP body 不能覆盖 operator identity），再加上通过的
  ≥72 小时、且完整绑定上述产物的 soak report。旧的手写
  `argus.soak-report.v1 {passed,duration_hours}` 不再接受；v2 必须包含带时区的
  起止时间、producer/build、原始 telemetry SHA、采样总数和连续性/零缺口证据。

可信 soak producer、原始 telemetry 哈希链和受保护密钥验签已经接入；但这不等于
已有一份通过的 72 小时证据。未按下述流程完整采集并成功 finalize 前，production
仍保持 NO-GO，不能手写或复制 report 中的 `passed` / `signature`。

## 可信 72 小时 soak

状态、API token 和 HMAC trust store 必须放在仓库外，并仅允许当前服务用户访问。
初始化一次 trust store（命令不会把密钥打印到终端）：

```bash
umask 077
install -d -m 700 /srv/.config/argus /srv/.local/state/argus
cd /srv/argus
.venv/bin/python - <<'PY'
import base64, json, os
from pathlib import Path

path = Path('/srv/.config/argus/soak-trust-store.json')
payload = {
    'schema_version': 'argus.soak-trust-store.v1',
    'keys': [{
        'key_id': 'gpu-validation-soak-2026-01',
        'algorithm': 'hmac-sha256',
        'secret_base64': base64.b64encode(os.urandom(32)).decode('ascii'),
        'enabled': True,
    }],
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write('\n')
PY
```

先用同一个工具在仓库外生成 `soak-subject.json`。它会重新计算 model、config、
profile、baseline、corpus 和 regression canonical JSON 摘要，并要求 regression report
内的 evaluated artifacts/runtime identity 完全一致；因此不要用文件 SHA 代替 canonical
JSON SHA，也不要手工猜测摘要：

```bash
cd /srv/argus
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python scripts/run_trusted_soak.py prepare-subject \
  --model-version-id <registry-model-version-id> \
  --camera-id validation_camera_01 \
  --registry-model-hash <registry-model-hash> \
  --model-path /srv/argus/data/exports/<version>/validation_camera_01/default/weights/torch/model.pt \
  --config-path /srv/argus/data/runtime/<version>.yaml \
  --scene-profile-path /srv/argus/data/scene_profiles/validation_camera_01/<approved-profile-id> \
  --baseline-path /srv/argus/data/baselines/<release-baseline> \
  --corpus-path /srv/argus-regression-capture \
  --regression-report /srv/argus/data/regression/reports/<version>/regression_report.json \
  --output /srv/.local/state/argus/soak-v3-001.subject.json
```

初始化新 run；目录已存在时命令会拒绝覆盖：

```bash
cd /srv/argus-validation/<run>/source
source /srv/argus-validation/<run>/state/validation.env
PYTHONPATH="$PWD/src:$PWD" /srv/argus/.venv/bin/python scripts/run_trusted_soak.py start \
  --state-dir /srv/argus-validation/<run>/state/soak-v3-001 \
  --subject /srv/argus-validation/<run>/state/soak-v3-001.subject.json \
  --base-url http://127.0.0.1:18081 \
  --token-file "$ARGUS_API_TOKEN_FILE" \
  --sample-interval 60 \
  --max-lateness 5
```

使用 systemd 最合适；临时验证可用下列命令持续采样。collector 使用单实例锁，
每个周期同时核对 Argus health、摄像头连接、SceneGuard/profile/model artifact、
摄像头帧与 SceneGuard 检查持续前进、source redirect 一致性、anomaly `torch-cuda`
状态和 `nvidia-smi` GPU identity，并以 canonical JSONL 哈希链
追加到外部状态目录。任何一次失败、漏采、机位阻断、身份变化或时钟异常都会令
本次 run 永久 NO-GO，应保留原始目录用于审计并新建 run，禁止修改 JSONL：

```bash
cd /srv/argus-validation/<run>/source
nohup env PYTHONPATH="$PWD/src:$PWD" /srv/argus/.venv/bin/python scripts/run_trusted_soak.py collect \
  --state-dir /srv/argus-validation/<run>/state/soak-v3-001 \
  --loop \
  > /srv/argus-validation/<run>/state/soak-v3-001.collector.log 2>&1 &

PYTHONPATH="$PWD/src:$PWD" /srv/argus/.venv/bin/python scripts/run_trusted_soak.py status \
  --state-dir /srv/argus-validation/<run>/state/soak-v3-001
```

连续采满至少 72 个 scheduled hours 后 collector 会输出 `complete=true`，但继续采样，
避免人工 finalize 前形成新的覆盖缺口。collector 有独立单实例锁；finalize 使用短时
状态锁，可以在 loop 运行时执行，成功写入 `finalized.json` 后 loop 会自动正常退出。
采样失败时 loop 会退出且 status 保持 `audit_valid=false`。

只有 `audit_valid=true`、`remaining_hours=0` 才尝试 finalize。finalize 会重新回放
完整哈希链；不足 72 小时、有 slot gap/覆盖缺口、wall/monotonic 回拨或漂移、重启
时间线不可能、SceneGuard/CUDA 降级、模型或 GPU identity 变化都会拒绝签名：
看到 `complete=true` 后应尽快 finalize；如果 collector 已异常退出并超过最后一个
采样周期，末端覆盖缺口会被拒绝，必须重新开始一轮完整 soak。

```bash
cd /srv/argus-validation/<run>/source
PYTHONPATH="$PWD/src:$PWD" /srv/argus/.venv/bin/python scripts/run_trusted_soak.py finalize \
  --state-dir /srv/argus-validation/<run>/state/soak-v3-001 \
  --trust-store /srv/.config/argus/soak-trust-store.json \
  --key-id gpu-validation-soak-2026-01 \
  --output /srv/argus-validation/<run>/state/soak-v3-001.report.json
```

发布请求中的 `soak_report_path` 指向签名 report，`soak_telemetry_path` 指向该 run
目录下的 `telemetry.jsonl`。服务端会再次比对 telemetry 文件 SHA、subject、72 小时
窗口、采样连续性和 HMAC；collector 日志本身不是发布证据。`ARGUS_SOAK_TRUST_STORE`
必须在 Argus 进程启动前进入其环境（systemd 部署写入 service 的 `Environment=`）；
在另一个 shell 临时 export 不会改变已运行进程，必须重启后再核对 health/model 状态。

任一字段缺失、使用 `approved: true` 简写、字符串/数字等 truthy 值、SHA
不匹配或指标不足时，canary → production 必须保持 NO-GO，且旧 production
不能被提前 retire。数据库保存 canonical evidence JSON 及其 SHA256；这能阻止
工作流误操作和发现事后篡改，但不是数字签名。能同时改数据库与 artifact 的
对抗者仍需 P3 的签名审批账本和受保护密钥解决。

`build_scene_profile_baseline.py` 默认仍使用 OpenCV 并生成 v2；该模式不能可靠区分
clean EOF 与解码错误，也没有可审计的 VFR PTS，因此只可用于离线训练。正式发布
baseline 必须显式传 `--strict-release-decoder`，由 FFprobe 与 FFmpeg rawvideo 解码器
生成 v3，并记录帧计数、单调 PTS、读取错误和 clean EOF。无论哪种模式，都不能
手工修改 `decoder_evidence` 或 `release_eligible` 绕过服务端复核。

注意：

- release-grade 回归固定 `--sample-every 1` 且不得使用 `--max-frames`。低频采样或截断
  只允许诊断；否则普通素材有效回放 `>=5 FPS` 和 projectile `>=59 FPS` 门禁必然失败。
- pipeline 模式的阈值、图像尺寸和后端来自配置文件；不要用 detector 模式参数代替生产配置。
- 发布口径固定为 `fp_gap_seconds=60`、`warmup_seconds=30`、普通素材有效回放
  `>=5 FPS`、projectile 源与有效回放均 `>=59 FPS`、SceneGuard coverage 门槛 `0.99`。
  CLI 覆盖只用于诊断；任一值偏离时 artifact binding 和 `--fail-on-acceptance`
  必须失败。运行时还应保存完整命令、配置副本及 SHA256。
- 隔离验证使用未获人工批准的 validation-only profile 时，发布门失败属于预期结果；不能以此产物替代正式批准。

截至 2026-07-15，`/srv/argus-regression-capture` 的已盘点语料仍只有约
0.5 小时 negative、10 个 static、10 个 projectile，projectile 为 30 FPS，且缺少完整
negative annotation、唯一 `source_session_id` 与 `independent_test` 证据。它低于
`>=2h / 20 / 20 / >=59 FPS` 门槛，所以只能做功能回归，不能启动 release-grade
72 小时 soak；production 必须保持 NO-GO。

确认 `/api/models/status` 中 anomaly backend 为 `torch-cuda`，不是 `openvino`、`torch-cpu` 或 `ssim-fallback`。
