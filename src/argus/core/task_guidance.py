"""Read-only, conservative task navigation; never a runtime or release authority.

Consumes the same public snapshots as the task API. It performs no IO and cannot
declare sampling, scene applicability, model quality or production readiness.
Snapshot agreement is an observation, not a cross-system transaction.
"""

from __future__ import annotations


def task_guidance(task: dict) -> dict:
    legacy = task.get("authority") == "legacy" and task.get("template_id") == "legacy_anomaly.v1"
    algorithm = task.get("template_id") == "fast_motion.v1"
    runtime = task.get("runtime") or {}
    source = task.get("source") or {}
    deployment = task.get("deployment") or {}
    result = {
        "schema_version": "argus.task-guidance.v1",
        "workflow_kind": "legacy_anomaly" if legacy else "algorithm" if algorithm else "model_task",
        "training_requirement": "legacy_workflow" if legacy else "not_applicable" if algorithm else "optional",
        "is_authorization": False,
        "production_scope": "existing_legacy_gates" if legacy else "unsupported",
    }

    def view(state: str, code: str, title: str, detail: str, target: str, action: str) -> dict:
        return {**result, "state": state, "reason_code": code, "title": title,
                "detail": detail, "target": target, "action_label": action}

    if task.get("lifecycle") == "archived":
        return view("archived", "TASK_ARCHIVED", "任务已归档", "保留历史，不再启动或修改此任务。",
                    "none", "")
    if runtime.get("state") in {"cleanup_pending", "cleanup_failed", "draining", "stopping"}:
        return view("attention", "CLEANUP_REQUIRED", "先确认旧运行实例已停止",
                    "清理未确认前不要重复启动。停止一个任务不应停止同源其它任务。",
                    "legacy_runtime" if legacy else "deployment", "查看停止与清理状态")
    if not legacy and deployment.get("runtime_status") in {"reconcile_required", "recovering"}:
        return view("attention", "RECONCILIATION_REQUIRED", "部署结果需要对账",
                    "保留原操作身份，核对实际应用回执；不要换请求重复部署或重新训练。",
                    "deployment", "查看部署与对账")
    if runtime.get("reason_code") == "SCENE_BLOCKED" or deployment.get("runtime_status") == "waiting_scene":
        return view("attention", "SCENE_BLOCKED", "场景条件尚未满足",
                    "检查机位、已批准 profile 和场景绑定。预览可用不表示检测已放行。",
                    "legacy_baseline" if legacy else "configuration", "检查场景与配置")
    if runtime.get("event_error") in {"EVENT_QUEUE_UNAVAILABLE", "EVENT_QUEUE_FAILED"}:
        return view("attention", "EVENT_EVIDENCE_UNAVAILABLE", "事件证据通道受阻",
                    "部分事件可能缺失。推理和预览独立显示，不能把没有事件视为正常。",
                    "events", "查看验证事件")
    if runtime.get("state") in {"failed", "runtime_dead", "blocked", "sampling_insufficient"}:
        detail = {
            "RESOURCE_EXHAUSTED": "资源不足；先检查现有任务和资源预算，不自动回退计算后端。",
            "SAMPLING_INSUFFICIENT": "实际采样不足；检查帧率与丢帧，不能把缺帧期间的无结果视为正常。",
            "LOAD_TIMEOUT": "模型加载超时；检查制品和运行环境，确认旧进程清理后再重试。",
            "INFER_TIMEOUT": "计算超时；先核对资源与清理状态，不接受旧执行的迟到结果。",
        }.get(runtime.get("reason_code"), "检查任务故障与运行依赖，确认清理完成后再按原部署流程恢复。")
        return view("attention", "RUNTIME_ATTENTION", "任务运行需要处理", detail,
                    "legacy_runtime" if legacy else "deployment", "查看运行故障")
    if source.get("state") != "running" or source.get("running") is not True or source.get("connected") is not True:
        unknown = (source.get("state", "unknown") == "unknown"
                   or source.get("running") is None or source.get("connected") is None)
        return view("unknown" if unknown else "attention", "SOURCE_UNVERIFIED" if unknown else "SOURCE_UNAVAILABLE",
                    "先核对摄像头源", "源状态尚未核实。检查连接与预览；此指引不会启动设备或执行模型。",
                    "source", "检查摄像头源")
    if legacy:
        if deployment.get("projection_status") != "authorized":
            return view("setup", "LEGACY_BINDING_REQUIRED", "核对场景、基线与模型绑定",
                        "按原流程完成正常性确认、训练、独立校准和受控绑定。历史快照不是当前放行证明。",
                        "legacy_baseline", "进入场景与基线流程")
        if runtime.get("state") == "running" and runtime.get("running") is True:
            return view("observed_running", "LEGACY_RUNTIME_OBSERVED", "已观测到异物任务运行",
                        "运行不等于场景已放行或精度验收通过；继续核对原检测链状态与告警证据。",
                        "legacy_runtime", "查看异物检测状态")
        return view("stopped" if runtime.get("state") == "stopped" else "unknown", "LEGACY_RUNTIME_UNVERIFIED",
                    "核对异物检测启动条件", "模型注册授权不等于实际加载；请通过原相机检测入口核对运行状态。",
                    "legacy_runtime", "查看异物检测状态")
    if task.get("authority") != "managed" or deployment.get("authority") != "managed":
        return view("unknown", "DEPLOYMENT_UNAVAILABLE", "任务部署身份尚未核实",
                    "无法确认部署管理权，请刷新并检查登记状态。", "deployment", "检查部署状态")
    if deployment.get("failure_code"):
        return view("attention", "DEPLOYMENT_FAILED", "最近部署操作未完成",
                    "旧版本可能仍在运行。先查看失败记录与实际应用版本，不要把旧运行视为新部署成功。",
                    "deployment", "查看部署记录")
    if deployment.get("runtime_status") in {"pending", "preparing", "starting", "applying"}:
        return view("transitioning", "DEPLOYMENT_PENDING", "部署请求正在处理",
                    "请求已登记不等于已应用；等待同一操作的回执，不重复提交。", "deployment", "查看部署进度")
    if deployment.get("desired_state") == "stopped":
        if runtime.get("running") is True or runtime.get("state") == "running":
            return view("attention", "UNEXPECTED_RUNTIME", "停止意图与运行观测不一致",
                        "请先核对停止与清理回执，不要直接重新启动。", "deployment", "核对停止结果")
        if runtime.get("state") != "stopped" or runtime.get("running") is not False:
            return view("unknown", "STOP_UNVERIFIED", "停止结果尚未核实",
                        "停止意图不是停止回执。先刷新运行观测并核对清理，不重复启动或修改配置。",
                        "deployment", "核对停止与清理")
        if not deployment.get("desired_bundle_id") and not deployment.get("applied_bundle_id"):
            return view("setup", "BUNDLE_SELECTION_REQUIRED", "配置并选择验证执行包",
                        "固定算法无需模型或训练。" if algorithm else "可以直接使用兼容的已有模型；只有需要定制时才进入采集与训练。",
                        "configuration" if algorithm else "models", "配置算法并冻结执行包" if algorithm else "选择模型与执行包")
        return view("stopped", "TASK_STOPPED", "任务已请求停止",
                    "核对停止回执后，可选择当前配置的执行包重新验证；历史已应用不代表当前仍在运行。",
                    "deployment", "查看部署与恢复")
    if deployment.get("desired_state") != "running":
        return view("unknown", "DEPLOYMENT_UNAVAILABLE", "部署意图尚未核实",
                    "请读取部署记录，不能根据摄像头运行推断任务运行。", "deployment", "检查部署状态")
    revision = deployment.get("deployment_revision")
    bundle = deployment.get("desired_bundle_id")
    matched = (
        isinstance(bundle, str) and bool(bundle) and bundle == deployment.get("applied_bundle_id") == runtime.get("bundle_id")
        and type(revision) is int and revision > 0
        and type(deployment.get("applied_deployment_revision")) is int
        and revision == deployment["applied_deployment_revision"]
        and type(runtime.get("deployment_revision")) is int and revision == runtime["deployment_revision"]
        and isinstance(deployment.get("current_operation_id"), str) and bool(deployment["current_operation_id"])
        and deployment["current_operation_id"] == runtime.get("deployment_operation_id")
        and type(task.get("current_revision")) is int and task["current_revision"] > 0
        and type(runtime.get("applied_task_revision")) is int
        and runtime["applied_task_revision"] == task.get("current_revision")
        and runtime.get("state") == "running" and (runtime.get("running") is None or runtime.get("running") is True)
        and deployment.get("runtime_status") == "running" and runtime.get("mode") == "shadow"
    )
    if not matched:
        return view("attention", "APPLICATION_UNVERIFIED", "实际应用版本尚未核实",
                    "任务配置、期望执行包和运行观测尚未完全一致。先查看部署与对账，不重复训练。",
                    "deployment", "核对实际应用版本")
    return view("observed_running", "SHADOW_RUNTIME_OBSERVED", "已观测到当前版本旁路运行",
                "这不是精度验收或生产授权。查看结果和证据；需要改善模型时再建立数据流程。",
                "events" if task.get("template_id") == "object_detection.v1" else "results", "查看验证事件" if task.get("template_id") == "object_detection.v1" else "查看旁路结果")
