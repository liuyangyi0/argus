import { ref, computed, onBeforeUnmount } from 'vue'
import { message } from 'ant-design-vue'
import { getStoryboard, getReplayMetadata } from '../api'
import type { StoryboardCamera } from '../types/api'

/**
 * Multi-camera synchronous replay controller.
 *
 * Owns a single "master clock" measured in seconds from the primary alert's
 * trigger_timestamp. Each storyboard camera maps its own video timeline as
 * `masterTime - camera.trigger_offset_s` (so a camera that triggered 300 ms
 * *after* the primary still lines up on the shared timeline).
 *
 * The controller is playback-authoritative. Individual StoryboardPlayer
 * components observe masterTime and advance their own <video>.currentTime —
 * they never drive the clock themselves. This keeps all cameras locked to
 * the same wall-clock moment even when one player stalls or buffers.
 */
export function useStoryboardController(alertId: string) {
  const cameras = ref<StoryboardCamera[]>([])
  const loading = ref(true)
  const error = ref<string | null>(null)

  // Per-camera total duration (seconds). The timeline scrubber spans the
  // widest camera so we can include all context before/after the primary.
  const durations = ref<Record<string, number>>({})

  // Master clock (seconds from primary alert's trigger).
  // 0 == every camera's trigger moment (modulo its trigger_offset_s).
  const masterTime = ref(0)
  const playing = ref(false)
  const speed = ref(1)

  // Scrubber bounds — computed from per-camera durations and offsets.
  // timelineStart is the earliest moment any camera has footage for;
  // timelineEnd is the latest.
  const timelineStart = computed(() => {
    if (cameras.value.length === 0) return 0
    let min = 0
    for (const cam of cameras.value) {
      // Camera local time 0 corresponds to master time cam.trigger_offset_s
      // (the camera's own trigger is at its local time = metadata.trigger_frame_index/fps).
      // We approximate clip start at masterTime = cam.trigger_offset_s - triggerOffsetInClip.
      // For simplicity we use "clip begins at 0 local time" -> master = cam.trigger_offset_s.
      const clipStart = cam.trigger_offset_s
      if (clipStart < min) min = clipStart
    }
    return min
  })

  const timelineEnd = computed(() => {
    if (cameras.value.length === 0) return 0
    let max = 0
    for (const cam of cameras.value) {
      const dur = durations.value[cam.alert_id] || 0
      const clipEnd = cam.trigger_offset_s + dur
      if (clipEnd > max) max = clipEnd
    }
    return max
  })

  /** Seconds duration of the shared timeline. Never negative. */
  const timelineDuration = computed(
    () => Math.max(0, timelineEnd.value - timelineStart.value),
  )

  /* ── Master clock — uses requestAnimationFrame with wall-clock deltas so
   *    playback is smooth and survives tab-suspend / speed changes. ──       */

  let rafId = 0
  let lastTs = 0

  function tick(now: number): void {
    if (!playing.value) {
      rafId = 0
      return
    }
    if (lastTs === 0) lastTs = now
    const deltaS = ((now - lastTs) / 1000) * speed.value
    lastTs = now
    const next = masterTime.value + deltaS
    if (next >= timelineEnd.value) {
      masterTime.value = timelineEnd.value
      playing.value = false
      rafId = 0
      return
    }
    masterTime.value = next
    rafId = requestAnimationFrame(tick)
  }

  function play(): void {
    if (playing.value) return
    if (cameras.value.length === 0) return
    // Seek master to start if we are at the end.
    if (masterTime.value >= timelineEnd.value - 0.05) {
      masterTime.value = timelineStart.value
    }
    playing.value = true
    lastTs = 0
    rafId = requestAnimationFrame(tick)
  }

  function pause(): void {
    playing.value = false
    lastTs = 0
    if (rafId) {
      cancelAnimationFrame(rafId)
      rafId = 0
    }
  }

  function togglePlay(): void {
    if (playing.value) pause()
    else play()
  }

  function seek(t: number): void {
    const clamped = Math.min(
      Math.max(t, timelineStart.value),
      timelineEnd.value,
    )
    masterTime.value = clamped
    // Restart raf loop on next frame so tick() picks up the new time cleanly.
    lastTs = 0
  }

  function setSpeed(s: number): void {
    speed.value = s
    // lastTs reset so the next tick doesn't double-count time.
    lastTs = 0
  }

  /** Report a camera's loaded video duration. Used for scrubber bounds. */
  function reportDuration(alertId: string, durationS: number): void {
    if (!isFinite(durationS) || durationS <= 0) return
    durations.value = { ...durations.value, [alertId]: durationS }
  }

  /* ── Data loading ── */

  async function load(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      const res = await getStoryboard(alertId)
      const list = (res?.cameras ?? []).slice(0, 4) // Hard cap
      cameras.value = list
      // Pre-fetch durations from metadata so the scrubber is accurate before
      // any <video> element has loaded. Failures are non-fatal.
      await Promise.all(
        list.map(async (c) => {
          try {
            const meta = await getReplayMetadata(c.alert_id)
            const fps = meta?.fps || 15
            const frames = meta?.frame_count || 0
            if (fps > 0 && frames > 0) {
              reportDuration(c.alert_id, frames / fps)
            }
          } catch (e) {
            // Swallow — the <video> will later report duration via reportDuration.
          }
        }),
      )
      // Default master time = 0 (primary trigger moment).
      masterTime.value = 0
    } catch (e) {
      const msg = e instanceof Error ? e.message : '多机位回放加载失败'
      error.value = msg
      message.error(msg)
      cameras.value = []
    } finally {
      loading.value = false
    }
  }

  /* ── Keyboard shortcuts (Space, ←, →) ── */

  function handleKeydown(e: KeyboardEvent): void {
    if (
      e.target instanceof HTMLInputElement ||
      e.target instanceof HTMLTextAreaElement ||
      e.target instanceof HTMLSelectElement
    ) return
    switch (e.key) {
      case ' ':
        e.preventDefault()
        togglePlay()
        break
      case 'ArrowLeft':
        e.preventDefault()
        seek(masterTime.value - 1)
        break
      case 'ArrowRight':
        e.preventDefault()
        seek(masterTime.value + 1)
        break
      case 'Home':
        seek(timelineStart.value)
        break
      case 'End':
        seek(timelineEnd.value)
        break
    }
  }

  /* ── Lifecycle ── */

  onBeforeUnmount(() => {
    pause()
  })

  return {
    // state
    cameras,
    loading,
    error,
    masterTime,
    playing,
    speed,
    durations,
    // computed
    timelineStart,
    timelineEnd,
    timelineDuration,
    // actions
    load,
    play,
    pause,
    togglePlay,
    seek,
    setSpeed,
    reportDuration,
    handleKeydown,
  }
}
