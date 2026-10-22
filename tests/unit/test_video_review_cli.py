"""CLI regression source; no CVAT server or FiftyOne database is needed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "video_review.py"
    spec = importlib.util.spec_from_file_location("argus_video_review_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _module(monkeypatch, name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _contracts(monkeypatch):
    bundle = object()
    save = Mock()
    _module(
        monkeypatch,
        "argus.integrations.contracts",
        load_bundle=Mock(return_value=bundle),
        save_bundle=save,
    )
    return bundle, save


def test_prepare_refuses_existing_output_before_preparing(cli, monkeypatch, tmp_path):
    _contracts(monkeypatch)
    prepare = Mock()
    _module(monkeypatch, "argus.integrations.corpus", prepare_review_bundle=prepare)
    output = tmp_path / "bundle.json"
    output.write_text("existing review", encoding="utf-8")

    assert cli.main([
        "prepare", "--manifest", str(tmp_path / "manifest.yaml"), "--output", str(output),
    ]) == 2

    prepare.assert_not_called()
    assert output.read_text(encoding="utf-8") == "existing review"


def test_overwrite_never_replaces_input_manifest(cli, monkeypatch, tmp_path):
    _contracts(monkeypatch)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("clips: []", encoding="utf-8")

    assert cli.main([
        "prepare", "--manifest", str(manifest), "--output", str(manifest), "--overwrite",
    ]) == 2
    assert manifest.read_text(encoding="utf-8") == "clips: []"


def test_push_uses_environment_token_and_durable_receipt(cli, monkeypatch, tmp_path):
    bundle, _ = _contracts(monkeypatch)
    push = Mock()
    _module(monkeypatch, "argus.integrations.cvat", export_bundle_to_cvat=push)
    monkeypatch.setenv("ARGUS_CVAT_TOKEN", "secret-token")
    monkeypatch.delenv("ARGUS_CVAT_USERNAME", raising=False)
    monkeypatch.delenv("ARGUS_CVAT_PASSWORD", raising=False)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"state": "ready"}', encoding="utf-8")

    assert cli.main([
        "cvat-push", "--bundle", str(tmp_path / "bundle.json"),
        "--url", "https://cvat.example", "--receipt", str(receipt),
    ]) == 0

    push.assert_called_once_with(
        bundle, url="https://cvat.example", receipt_path=receipt,
        username=None, password=None, token="secret-token", organization=None,
    )


def test_push_refuses_receipt_that_is_input_bundle(cli, monkeypatch, tmp_path):
    _contracts(monkeypatch)
    push = Mock()
    _module(monkeypatch, "argus.integrations.cvat", export_bundle_to_cvat=push)
    bundle = str(tmp_path / "bundle.json")
    assert cli.main([
        "cvat-push", "--bundle", bundle, "--url", "https://cvat.example", "--receipt", bundle,
    ]) == 2
    push.assert_not_called()


@pytest.mark.parametrize("url", [
    "https://user:pass@cvat.example", "https://cvat.example?token=secret",
    "file:///tmp/cvat", "https://cvat.example#token",
])
def test_credentials_never_go_in_url(cli, url):
    with pytest.raises(ValueError, match="without credentials"):
        cli._cvat_credentials(url)


def test_cli_does_not_accept_secret_arguments(cli, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args([
            "cvat-push", "--bundle", "bundle.json", "--url", "https://cvat.example",
            "--receipt", "receipt.json", "--token", "secret",
        ])
    assert exc.value.code == 2
    assert "secret" not in capsys.readouterr().err


@pytest.mark.parametrize("secret_args", [
    ["--password", "sensitive value with spaces"],
    ["--password=sensitive-value"],
    ["--access-token", "sensitive-value"],
])
def test_rejected_secret_arguments_are_not_echoed(cli, capsys, secret_args):
    with pytest.raises(SystemExit):
        cli.parse_args([
            "cvat-push", "--bundle", "bundle.json", "--url", "https://cvat.example",
            "--receipt", "receipt.json", *secret_args,
        ])
    error = capsys.readouterr().err
    assert "sensitive" not in error
    assert "[redacted]" in error


def test_pull_passes_reviewer_and_explicit_empty_normal_confirmations(cli, monkeypatch, tmp_path):
    bundle, save = _contracts(monkeypatch)
    reviewed = object()
    pull = Mock(return_value=reviewed)
    _module(monkeypatch, "argus.integrations.cvat", import_bundle_from_cvat=pull)
    monkeypatch.setenv("ARGUS_CVAT_USERNAME", "review-user")
    monkeypatch.setenv("ARGUS_CVAT_PASSWORD", "review-password")
    monkeypatch.delenv("ARGUS_CVAT_TOKEN", raising=False)
    output = tmp_path / "reviewed.json"

    assert cli.main([
        "cvat-pull", "--bundle", str(tmp_path / "bundle.json"),
        "--url", "https://cvat.example", "--receipt", str(tmp_path / "receipt.json"),
        "--reviewed-by", "operator-42", "--confirm-normal-clip", "negative-01",
        "--confirm-normal-clip", "negative-02", "--output", str(output),
    ]) == 0

    assert pull.call_args.args == (bundle,)
    assert pull.call_args.kwargs["reviewed_by"] == "operator-42"
    assert pull.call_args.kwargs["confirmed_normal_clip_ids"] == ["negative-01", "negative-02"]
    save.assert_called_once_with(reviewed, output, overwrite=False)


def test_pull_checks_output_before_contacting_cvat(cli, monkeypatch, tmp_path):
    _contracts(monkeypatch)
    pull = Mock()
    _module(monkeypatch, "argus.integrations.cvat", import_bundle_from_cvat=pull)
    output = tmp_path / "reviewed.json"
    output.write_text("existing", encoding="utf-8")

    assert cli.main([
        "cvat-pull", "--bundle", str(tmp_path / "bundle.json"),
        "--url", "https://cvat.example", "--receipt", str(tmp_path / "receipt.json"),
        "--reviewed-by", "operator-42", "--output", str(output),
    ]) == 2
    pull.assert_not_called()


def test_pull_output_cannot_create_a_permanent_receipt_lock(cli, monkeypatch, tmp_path):
    _contracts(monkeypatch)
    pull = Mock()
    _module(monkeypatch, "argus.integrations.cvat", import_bundle_from_cvat=pull)
    receipt = tmp_path / "receipt.json"
    lock = tmp_path / "receipt.json.lock"
    assert cli.main([
        "cvat-pull", "--bundle", str(tmp_path / "bundle.json"),
        "--url", "https://cvat.example", "--receipt", str(receipt),
        "--reviewed-by", "reviewer", "--output", str(lock),
    ]) == 2
    pull.assert_not_called()
    assert not lock.exists()


def test_prepare_output_cannot_overwrite_manifest_bound_evaluation(cli, monkeypatch, tmp_path):
    report = tmp_path / "pipeline-evaluation.json"
    report.write_text("original report", encoding="utf-8")
    bundle = SimpleNamespace(clips=[SimpleNamespace(evaluation_path=str(report), clip_id="clip")])
    save = Mock()
    _module(monkeypatch, "argus.integrations.contracts", load_bundle=Mock(), save_bundle=save)
    _module(monkeypatch, "argus.integrations.corpus", prepare_review_bundle=Mock(return_value=bundle))
    assert cli.main([
        "prepare", "--manifest", str(tmp_path / "manifest.yaml"),
        "--output", str(report), "--overwrite",
    ]) == 2
    save.assert_not_called()
    assert report.read_text(encoding="utf-8") == "original report"


def test_overwrite_pull_cannot_replace_source_media(cli, monkeypatch, tmp_path):
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"source video")
    bundle = SimpleNamespace(clips=[SimpleNamespace(video_path=str(source))])
    _module(
        monkeypatch, "argus.integrations.contracts",
        load_bundle=Mock(return_value=bundle), save_bundle=Mock(),
    )
    pull = Mock()
    _module(monkeypatch, "argus.integrations.cvat", import_bundle_from_cvat=pull)

    assert cli.main([
        "cvat-pull", "--bundle", str(tmp_path / "bundle.json"),
        "--url", "https://cvat.example", "--receipt", str(tmp_path / "receipt.json"),
        "--reviewed-by", "operator-42", "--output", str(source), "--overwrite",
    ]) == 2
    pull.assert_not_called()
    assert source.read_bytes() == b"source video"


def test_export_labels_preserves_explicit_split(cli, monkeypatch, tmp_path):
    bundle, _ = _contracts(monkeypatch)
    output = tmp_path / "test-labels"
    export = Mock(return_value=output / "manifest.json")
    _module(monkeypatch, "argus.integrations.corpus", export_review_labels=export)

    assert cli.main([
        "export-labels", "--bundle", str(tmp_path / "reviewed.json"),
        "--split", "test", "--output-dir", str(output),
    ]) == 0
    export.assert_called_once_with(
        bundle, output, overwrite=False, split="test",
        protected_paths=[tmp_path / "reviewed.json"],
    )


def test_overwrite_export_cannot_replace_input_bundle(cli, monkeypatch, tmp_path):
    _contracts(monkeypatch)
    export = Mock()
    _module(monkeypatch, "argus.integrations.corpus", export_review_labels=export)

    assert cli.main([
        "export-labels", "--bundle", str(tmp_path / "manifest.json"),
        "--output-dir", str(tmp_path), "--overwrite",
    ]) == 2
    export.assert_not_called()


def test_missing_sdk_reports_targeted_install_without_disclosing_credentials(
    cli, monkeypatch, capsys,
):
    def fail(_args):
        raise ModuleNotFoundError("No module named cvat_sdk", name="cvat_sdk")

    monkeypatch.setattr(cli, "run", fail)
    assert cli.main([
        "cvat-push", "--bundle", "bundle.json", "--url", "https://cvat.example",
        "--receipt", "receipt.json",
    ]) == 2
    assert 'cvat-sdk>=2.79,<2.80' in capsys.readouterr().err


def test_errors_redact_environment_credentials(cli, monkeypatch, capsys):
    def fail(_args):
        raise RuntimeError("server echoed secret-token and private-password")

    monkeypatch.setenv("ARGUS_CVAT_TOKEN", "secret-token")
    monkeypatch.setenv("ARGUS_CVAT_PASSWORD", "private-password")
    monkeypatch.setattr(cli, "run", fail)
    assert cli.main(["fiftyone-import", "--bundle", "bundle.json", "--dataset-name", "review"]) == 2
    error = capsys.readouterr().err
    assert "secret-token" not in error
    assert "private-password" not in error
    assert "[redacted]" in error


def test_redaction_masks_long_secrets_before_overlapping_usernames(cli, monkeypatch):
    monkeypatch.setenv("ARGUS_CVAT_USERNAME", "prefix")
    monkeypatch.setenv("ARGUS_CVAT_TOKEN", "prefix-private-suffix")
    monkeypatch.setenv("ARGUS_CVAT_ACCESS_TOKEN", "alias-access-token")
    message = cli._safe_error(RuntimeError("prefix-private-suffix alias-access-token"))
    assert message == "[redacted] [redacted]"


@pytest.mark.parametrize("uri", [
    "mongodb://review-user:review-password@db.invalid/review?token=query-secret",
    "mongodb+srv://review-user:review-password@cluster.invalid/review?token=query-secret",
    "MongoDB://review-user:review-password@db.invalid/review?token=query-secret",
    "mongodb://review-user:review'password@db.invalid/review?token=query-secret",
])
def test_mongodb_config_uris_are_redacted_without_environment_credentials(cli, monkeypatch, uri):
    for name in cli._SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    # A short existing username must not destroy the URI scheme before masking.
    monkeypatch.setenv("ARGUS_CVAT_USERNAME", "mongo")
    message = cli._safe_error(RuntimeError(
        f"ValueError: source refused; cleanup OperationFailure at '{uri}' ; lock retained"
    ))
    assert "ValueError: source refused" in message
    assert "OperationFailure" in message and "lock retained" in message
    for secret in ("review-user", "review-password", "review'password", "query-secret"):
        assert secret not in message
    assert "[redacted]" in message


def test_combined_import_and_lock_failure_diagnostics_are_redacted_on_stderr(
    cli, monkeypatch, capsys,
):
    uri = "mongodb://review-user:review-password@db.invalid/review?token=query-secret"
    for name in cli._SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIFTYONE_DATABASE_URI", uri)
    failure = RuntimeError(
        "import failed (ValueError: image artifact changed); "
        f"lock release unconfirmed (OperationFailure at {uri}) ; administrator recovery required"
    )
    monkeypatch.setattr(cli, "run", Mock(side_effect=failure))
    assert cli.main([
        "scene-images-import", "--bundle", "bundle.json", "--manifest", "images/manifest.json",
        "--scene-comparison", "report.json", "--dataset-name", "review",
    ]) == 2
    error = capsys.readouterr().err
    assert "image artifact changed" in error and "lock release unconfirmed" in error
    assert "administrator recovery required" in error
    assert "review-password" not in error and "query-secret" not in error
    assert "[redacted]" in error


def test_real_lock_failure_reports_primary_refusal_and_cleanup_type(cli, monkeypatch, capsys):
    from argus.integrations.fiftyone_scene_images import ImportLockFailure

    for name in cli._SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    primary = ValueError("image artifact changed while importing")
    cleanup = ConnectionError("mongodb://review-user:review-password@db.invalid/review")
    failure = ImportLockFailure(
        "image dataset database import lock release is unconfirmed; administrator recovery required",
        import_error=primary, cleanup_error=cleanup,
    )
    monkeypatch.setattr(cli, "run", Mock(side_effect=failure))
    assert cli.main([
        "scene-images-import", "--bundle", "bundle.json", "--manifest", "images/manifest.json",
        "--scene-comparison", "report.json", "--dataset-name", "review",
    ]) == 2
    error = capsys.readouterr().err
    assert "ValueError: image artifact changed while importing" in error
    assert "cleanup error=ConnectionError" in error
    assert "lock release is unconfirmed" in error
    assert "review-password" not in error
    assert failure.import_error is primary and failure.cleanup_error is cleanup


@pytest.mark.parametrize("failure_mode", ["raises", "non-string", "interrupt"])
def test_unprintable_sdk_error_still_returns_controlled_failure_without_secondary_details(
    cli, monkeypatch, capsys, failure_mode,
):
    calls = []

    class UnprintableSdkError(RuntimeError):
        def __str__(self):
            calls.append("str")
            if failure_mode == "raises":
                raise AssertionError("secret diagnostic must not escape error reporting")
            if failure_mode == "interrupt":
                raise KeyboardInterrupt("secret interrupted diagnostic")
            return None

        def __repr__(self):
            raise AssertionError("error reporting must not fall back to SDK repr")

    failure = UnprintableSdkError("private credentials in SDK args")
    monkeypatch.setattr(cli, "run", Mock(side_effect=failure))
    assert cli.main([
        "scene-images-import", "--bundle", "bundle.json", "--manifest", "images/manifest.json",
        "--scene-comparison", "report.json", "--dataset-name", "review",
    ]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "UnprintableSdkError" in output.err and "details unavailable" in output.err
    assert "secret" not in output.err and "private credentials" not in output.err
    assert calls == ["str"]


@pytest.mark.parametrize("launch", [False, True])
def test_fiftyone_launch_is_explicit_and_waits(cli, monkeypatch, launch):
    bundle, _ = _contracts(monkeypatch)
    view = object()
    load_view = Mock(return_value=view)
    dataset = SimpleNamespace(load_saved_view=load_view)
    import_dataset = Mock(return_value=dataset)
    _module(monkeypatch, "argus.integrations.fiftyone", import_bundle_to_fiftyone=import_dataset)
    wait = Mock()
    app = Mock(return_value=SimpleNamespace(wait=wait))
    _module(monkeypatch, "fiftyone", launch_app=app)
    argv = ["fiftyone-import", "--bundle", "bundle.json", "--dataset-name", "review"]
    if launch:
        argv.append("--launch")

    assert cli.main(argv) == 0
    import_dataset.assert_called_once_with(bundle, dataset_name="review", persistent=True)
    if launch:
        load_view.assert_called_once_with("argus-current")
        app.assert_called_once_with(view=view)
        wait.assert_called_once_with()
    else:
        load_view.assert_not_called()
        app.assert_not_called()
        wait.assert_not_called()


def test_fiftyone_scene_comparison_option_is_repeatable(cli):
    args = cli.parse_args([
        "fiftyone-import", "--bundle", "bundle.json", "--dataset-name", "review",
        "--scene-comparison", "scene-one.json", "--scene-comparison", "scene-two.json",
    ])
    assert args.scene_comparison == [Path("scene-one.json"), Path("scene-two.json")]


@pytest.mark.parametrize("launch", [False, True])
def test_fiftyone_attaches_validated_scene_reports_and_launches_disagreement_view(
    cli, monkeypatch, launch, capsys,
):
    bundle, _ = _contracts(monkeypatch)
    scene_mapping = {"clip-one": object(), "clip-two": object()}
    load_comparisons = Mock(return_value=scene_mapping)
    _module(
        monkeypatch, "argus.integrations.scene_comparison",
        load_scene_comparisons=load_comparisons,
    )
    view = object()
    load_view = Mock(return_value=view)
    dataset = SimpleNamespace(load_saved_view=load_view)
    import_dataset = Mock(return_value=dataset)
    _module(monkeypatch, "argus.integrations.fiftyone", import_bundle_to_fiftyone=import_dataset)
    wait = Mock()
    app = Mock(return_value=SimpleNamespace(wait=wait))
    _module(monkeypatch, "fiftyone", launch_app=app)
    argv = [
        "fiftyone-import", "--bundle", "bundle.json", "--dataset-name", "review",
        "--scene-comparison", "scene-one.json", "--scene-comparison", "scene-two.json",
    ]
    if launch:
        argv.append("--launch")

    assert cli.main(argv) == 0
    load_comparisons.assert_called_once_with(
        bundle, [Path("scene-one.json"), Path("scene-two.json")],
    )
    import_dataset.assert_called_once_with(
        bundle, dataset_name="review", persistent=True, scene_comparisons=scene_mapping,
    )
    output = capsys.readouterr().out
    assert "reports attached: 2" in output
    assert "argus-scene-raw-disagreements" in output
    assert "argus-scene-errors" in output
    assert "an empty disagreement view is not a pass" in output
    if launch:
        load_view.assert_called_once_with("argus-scene-disagreements")
        app.assert_called_once_with(view=view)
        wait.assert_called_once_with()
    else:
        load_view.assert_not_called()
        app.assert_not_called()
        wait.assert_not_called()


def test_fiftyone_rejects_invalid_scene_report_before_importing_dataset(
    cli, monkeypatch, capsys,
):
    _contracts(monkeypatch)
    load_comparisons = Mock(side_effect=ValueError("scene report timestamp mismatch"))
    _module(
        monkeypatch, "argus.integrations.scene_comparison",
        load_scene_comparisons=load_comparisons,
    )
    import_dataset = Mock()
    _module(monkeypatch, "argus.integrations.fiftyone", import_bundle_to_fiftyone=import_dataset)

    assert cli.main([
        "fiftyone-import", "--bundle", "bundle.json", "--dataset-name", "review",
        "--scene-comparison", "invalid-scene.json", "--launch",
    ]) == 2
    import_dataset.assert_not_called()
    assert "scene report timestamp mismatch" in capsys.readouterr().err


def test_fiftyone_empty_scene_mapping_still_uses_scene_review_view(cli, monkeypatch):
    bundle, _ = _contracts(monkeypatch)
    _module(
        monkeypatch, "argus.integrations.scene_comparison",
        load_scene_comparisons=Mock(return_value={}),
    )
    load_view = Mock(return_value=object())
    import_dataset = Mock(return_value=SimpleNamespace(load_saved_view=load_view))
    _module(monkeypatch, "argus.integrations.fiftyone", import_bundle_to_fiftyone=import_dataset)
    _module(monkeypatch, "fiftyone", launch_app=Mock(return_value=SimpleNamespace(wait=Mock())))

    assert cli.main([
        "fiftyone-import", "--bundle", "bundle.json", "--dataset-name", "review",
        "--scene-comparison", "no-disagreements.json", "--launch",
    ]) == 0
    import_dataset.assert_called_once_with(
        bundle, dataset_name="review", persistent=True, scene_comparisons={},
    )
    load_view.assert_called_once_with("argus-scene-disagreements")


@pytest.mark.parametrize("command,extra", [
    ("scene-images-prepare", ["--output-dir", "images"]),
    ("scene-images-import", ["--manifest", "images/manifest.json", "--dataset-name", "images"]),
])
def test_scene_images_require_explicit_reports(cli, command, extra):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args([command, "--bundle", "bundle.json", *extra])
    assert exc.value.code == 2


def test_scene_images_prepare_defaults_and_repeatable_reports(cli):
    args = cli.parse_args([
        "scene-images-prepare", "--bundle", "bundle.json", "--output-dir", "images",
        "--scene-comparison", "first.json", "--scene-comparison", "second.json",
    ])
    assert args.scene_comparison == [Path("first.json"), Path("second.json")]
    assert args.selection == "disagreements"
    assert args.max_images == 10000
    assert args.max_output_bytes == 10737418240


@pytest.mark.parametrize("flag,value", [
    ("--max-images", "0"), ("--max-images", "-1"), ("--max-images", "1.5"),
    ("--max-output-bytes", "0"), ("--max-output-bytes", "-20"),
])
def test_scene_images_reject_invalid_budgets_before_running(cli, flag, value):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args([
            "scene-images-prepare", "--bundle", "bundle.json", "--output-dir", "images",
            "--scene-comparison", "scene.json", flag, value,
        ])
    assert exc.value.code == 2


def test_scene_images_prepare_has_no_overwrite_switch(cli):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args([
            "scene-images-prepare", "--bundle", "bundle.json", "--output-dir", "images",
            "--scene-comparison", "scene.json", "--overwrite",
        ])
    assert exc.value.code == 2


def test_scene_images_prepare_passes_explicit_limits_and_protects_bundle(
    cli, monkeypatch, tmp_path, capsys,
):
    bundle, _ = _contracts(monkeypatch)
    comparisons = {"clip-one": object()}
    load = Mock(return_value=comparisons)
    _module(
        monkeypatch, "argus.integrations.scene_comparison", load_scene_comparisons=load,
    )
    output = tmp_path / "images-generation-b"
    prepare = Mock(return_value=output / "manifest.json")
    _module(
        monkeypatch, "argus.integrations.scene_review_images",
        prepare_scene_review_images=prepare,
    )
    argv = [
        "scene-images-prepare", "--bundle", str(tmp_path / "bundle.json"),
        "--scene-comparison", "scene.json", "--output-dir", str(output),
        "--selection", "all-compared", "--max-images", "300", "--max-output-bytes", "4096",
    ]
    assert cli.main(argv) == 0
    load.assert_called_once_with(bundle, [Path("scene.json")])
    prepare.assert_called_once_with(
        bundle, scene_comparisons=comparisons, output_dir=output,
        selection="all-compared", max_images=300, max_output_bytes=4096,
        protected_paths=[tmp_path / "bundle.json"],
    )
    assert "Diagnostic images only" in capsys.readouterr().out


@pytest.mark.parametrize("command,extra,module_name,function_name", [
    (
        "scene-images-prepare", ["--output-dir", "images"],
        "argus.integrations.scene_review_images", "prepare_scene_review_images",
    ),
    (
        "scene-images-import", ["--manifest", "images/manifest.json", "--dataset-name", "images"],
        "argus.integrations.fiftyone_scene_images", "import_scene_review_images",
    ),
])
def test_scene_images_validate_reports_before_decode_or_sdk(
    cli, monkeypatch, command, extra, module_name, function_name,
):
    _contracts(monkeypatch)
    _module(
        monkeypatch, "argus.integrations.scene_comparison",
        load_scene_comparisons=Mock(side_effect=ValueError("report PTS mismatch")),
    )
    operation = Mock()
    _module(monkeypatch, module_name, **{function_name: operation})
    assert cli.main([
        command, "--bundle", "bundle.json", "--scene-comparison", "invalid.json", *extra,
    ]) == 2
    operation.assert_not_called()


@pytest.mark.parametrize("launch", [False, True])
def test_scene_images_import_uses_image_adapter_and_unreviewed_view(
    cli, monkeypatch, launch, capsys,
):
    bundle, _ = _contracts(monkeypatch)
    comparisons = {"clip-one": object()}
    load = Mock(return_value=comparisons)
    _module(
        monkeypatch, "argus.integrations.scene_comparison", load_scene_comparisons=load,
    )
    view = object()
    load_view = Mock(return_value=view)
    dataset = SimpleNamespace(load_saved_view=load_view)
    importer = Mock(return_value=dataset)
    _module(
        monkeypatch, "argus.integrations.fiftyone_scene_images", import_scene_review_images=importer,
    )
    video_importer = Mock()
    _module(
        monkeypatch, "argus.integrations.fiftyone", import_bundle_to_fiftyone=video_importer,
    )
    wait = Mock()
    launch_app = Mock(return_value=SimpleNamespace(wait=wait))
    _module(monkeypatch, "fiftyone", launch_app=launch_app)
    argv = [
        "scene-images-import", "--bundle", "bundle.json", "--manifest", "images/manifest.json",
        "--scene-comparison", "scene.json", "--dataset-name", "review-images",
    ]
    if launch:
        argv.append("--launch")
    assert cli.main(argv) == 0
    load.assert_called_once_with(bundle, [Path("scene.json")])
    importer.assert_called_once_with(
        bundle, manifest_path=Path("images/manifest.json"), scene_comparisons=comparisons,
        dataset_name="review-images", persistent=True,
    )
    video_importer.assert_not_called()
    assert "not a pass or release approval" in capsys.readouterr().out
    if launch:
        load_view.assert_called_once_with("argus-scene-unreviewed")
        launch_app.assert_called_once_with(view=view)
        wait.assert_called_once_with()
    else:
        load_view.assert_not_called()
        launch_app.assert_not_called()
        wait.assert_not_called()


def test_scene_image_import_failure_does_not_launch_app(cli, monkeypatch):
    _contracts(monkeypatch)
    _module(
        monkeypatch, "argus.integrations.scene_comparison", load_scene_comparisons=Mock(return_value={}),
    )
    _module(
        monkeypatch, "argus.integrations.fiftyone_scene_images",
        import_scene_review_images=Mock(side_effect=ValueError("PNG digest mismatch")),
    )
    launch_app = Mock()
    _module(monkeypatch, "fiftyone", launch_app=launch_app)
    assert cli.main([
        "scene-images-import", "--bundle", "bundle.json", "--manifest", "images/manifest.json",
        "--scene-comparison", "scene.json", "--dataset-name", "review-images", "--launch",
    ]) == 2
    launch_app.assert_not_called()
