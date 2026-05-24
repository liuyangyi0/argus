# Camera Input Migration TODO

## Status Legend

- `[ ]` TODO
- `[~]` IN_PROGRESS
- `[x]` DONE
- `[!]` BLOCKED

## Wave 0: Baseline

- [x] Inspect current camera/input implementation and worktree status
- [x] Run current targeted tests
- [x] Record current API compatibility requirements

## Wave 1: Runtime Planning

- [x] Agent A: CameraRuntimePlan / CameraRuntimePlanner
- [x] Agent B: StreamRegistry
- [x] Agent C: Dashboard API contract tests

## Wave 2: Execution Boundaries

- [x] Agent D: CaptureFactory / DetectionPipeline adapter
- [x] Agent E: PreviewGateway
- [x] Agent F: CameraOrchestrator / Bootstrap

## Wave 3: Cleanup

- [!] Remove source_resolver compatibility path - deferred as compatibility shim for this migration
- [!] Remove dashboard direct go2rtc calls - partially deferred for compatibility wrappers
- [!] Remove pipeline protocol branching - deferred until capture factory is fully adopted by CameraManager
- [x] Run targeted tests
- [x] Update architecture docs
