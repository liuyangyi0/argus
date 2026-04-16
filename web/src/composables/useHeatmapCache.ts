import { ref, onBeforeUnmount } from 'vue'
import { getReplayHeatmapUrl } from '../api'

/**
 * Preloads and caches all heatmap frames as ImageBitmap objects
 * for instant GPU-to-GPU drawing on the compositor canvas.
 */
export function useHeatmapCache(alertId: string, frameCount: number) {
  const cache = ref<Map<number, ImageBitmap>>(new Map())
  const loading = ref(false)
  const progress = ref(0) // 0..1

  const BATCH_SIZE = 10
  const BATCH_DELAY_MS = 50
  let disposed = false

  async function fetchFrame(index: number): Promise<void> {
    if (disposed) return
    const url = getReplayHeatmapUrl(alertId, index)
    try {
      const res = await fetch(url)
      if (!res.ok) return // 404 = sparse frame, skip
      const blob = await res.blob()
      if (disposed) return
      const bitmap = await createImageBitmap(blob)
      if (disposed) {
        bitmap.close()
        return
      }
      cache.value.set(index, bitmap)
    } catch {
      // Network error or decode failure — skip this frame
    }
  }

  async function startPreloading(): Promise<void> {
    if (frameCount <= 0) return
    loading.value = true
    progress.value = 0

    let loaded = 0
    for (let batchStart = 0; batchStart < frameCount; batchStart += BATCH_SIZE) {
      if (disposed) break
      const batchEnd = Math.min(batchStart + BATCH_SIZE, frameCount)
      const promises: Promise<void>[] = []
      for (let i = batchStart; i < batchEnd; i++) {
        promises.push(fetchFrame(i))
      }
      await Promise.all(promises)
      loaded += batchEnd - batchStart
      progress.value = loaded / frameCount
      if (batchEnd < frameCount && !disposed) {
        await new Promise((r) => setTimeout(r, BATCH_DELAY_MS))
      }
    }

    loading.value = false
    progress.value = 1
  }

  function getFrame(index: number): ImageBitmap | null {
    return cache.value.get(index) ?? null
  }

  function dispose(): void {
    disposed = true
    for (const bitmap of cache.value.values()) {
      bitmap.close()
    }
    cache.value.clear()
  }

  // Start preloading immediately
  startPreloading()

  onBeforeUnmount(() => {
    dispose()
  })

  return {
    cache,
    loading,
    progress,
    getFrame,
    dispose,
  }
}
