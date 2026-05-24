# Camera Input Runtime Architecture

This document records the camera input migration boundary after the runtime
planning refactor.

## Core Flow

```text
CameraConfig
  -> CameraRuntimePlanner
  -> CameraRuntimePlan
  -> StreamRegistry / PreviewGateway / CaptureFactory / CameraOrchestrator
```

`CameraConfig.source` and `CameraConfig.protocol` remain persisted user intent.
They must not be overwritten with runtime relay URLs.

## Responsibilities

- `argus.camera.planner` builds immutable runtime plans from original camera
  config.
- `argus.streaming.stream_registry` owns desired go2rtc streams and reconciles
  them after startup or restart.
- `argus.streaming.preview_gateway` owns browser preview URLs, snapshot JPEG,
  MJPEG latest-frame streams, heatmap streams, and running-camera connection
  probes.
- `argus.capture.factory` creates capture adapters from runtime detection input
  while the pipeline still keeps its legacy fallback branch.
- `argus.camera.orchestrator` coordinates camera lifecycle calls with stream
  registration.

## Protocol Defaults

- USB: go2rtc owns the device; detection reads the go2rtc RTSP relay.
- RTSP: detection reads the original RTSP URL; preview uses go2rtc relay.
- GigE: detection uses the GigE SDK; optional exec preview remains supported.
- File: detection reads the file; preview uses the pipeline latest frame.

## Compatibility Shims

The migration intentionally keeps these shims for now:

- `CameraManager.source_resolver`
- `CameraSourceResolution`
- `go2rtc_manager` compatibility functions
- pipeline protocol branching when no injected capture adapter is provided

These can be removed in a later cleanup once all callers use
`CameraRuntimePlan`, `StreamRegistry`, and `CaptureFactory` directly.
