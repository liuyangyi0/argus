import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  getBaselines,
  getBaselineVersions,
  getCameraGroups,
  verifyBaseline as apiVerifyBaseline,
  activateBaseline as apiActivateBaseline,
  retireBaseline as apiRetireBaseline,
  deleteBaselineVersion as apiDeleteBaselineVersion,
  mergeGroupBaseline as apiMergeGroupBaseline,
  mergeFalsePositives as apiMergeFalsePositives,
  optimizeBaseline as apiOptimizeBaseline,
  previewOptimize as apiPreviewOptimize,
} from '../api'
import type { BaselineInfo, BaselineVersionInfo } from '../types/api'

interface CameraGroup {
  group_id: string
  name: string
  camera_ids: string[]
  image_count: number
  current_version: string | null
}

/**
 * Pinia store for baseline management state.
 *
 * Owns the data layer (API calls, loading flags, drawer visibility) so that
 * BaselineTable / BaselineVersionDrawer / BaselineCaptureModals can share
 * state without heavy prop drilling. UI-level concerns (Modal.confirm,
 * message.success) stay in the components that render the buttons.
 */
export const useBaselineStore = defineStore('baselineStore', () => {
  // ── Baselines ──
  const baselines = ref<BaselineInfo[]>([])
  const baselinesLoading = ref(false)

  async function loadBaselines() {
    baselinesLoading.value = true
    try {
      const res = await getBaselines()
      baselines.value = res.baselines || []
    } finally {
      baselinesLoading.value = false
    }
  }

  // ── Baseline versions (for version drawer) ──
  const baselineVersions = ref<BaselineVersionInfo[]>([])
  const versionsLoading = ref(false)
  const versionDrawerVisible = ref(false)
  const versionDrawerCamera = ref('')

  async function loadBaselineVersions(cameraId: string) {
    versionsLoading.value = true
    try {
      const res = await getBaselineVersions({ camera_id: cameraId })
      baselineVersions.value = res.versions || []
    } catch {
      baselineVersions.value = []
      throw new Error('加载基线版本失败')
    } finally {
      versionsLoading.value = false
    }
  }

  function openVersionDrawer(cameraId: string) {
    versionDrawerCamera.value = cameraId
    versionDrawerVisible.value = true
    return loadBaselineVersions(cameraId)
  }

  function closeVersionDrawer() {
    versionDrawerVisible.value = false
  }

  // ── Camera groups ──
  const cameraGroups = ref<CameraGroup[]>([])
  const groupsLoading = ref(false)

  async function loadCameraGroups() {
    groupsLoading.value = true
    try {
      const res = await getCameraGroups()
      cameraGroups.value = res.groups || []
    } finally {
      groupsLoading.value = false
    }
  }

  // ── Per-item loading flags (for button spinners) ──
  const verifyingVersion = ref<string | null>(null)
  const activatingVersion = ref<string | null>(null)
  const deletingBaseline = ref<string | null>(null)
  const mergingGroup = ref<string | null>(null)
  const mergingFP = ref<string | null>(null)
  const optimizingBaseline = ref<string | null>(null)

  // ── Version workflow actions ──
  async function verifyBaselineVersion(cameraId: string, version: string, verifiedBy: string) {
    verifyingVersion.value = version
    try {
      await apiVerifyBaseline({ camera_id: cameraId, version, verified_by: verifiedBy })
      await loadBaselineVersions(cameraId)
      await loadBaselines()
    } finally {
      verifyingVersion.value = null
    }
  }

  async function activateBaselineVersion(cameraId: string, version: string) {
    activatingVersion.value = version
    try {
      await apiActivateBaseline({ camera_id: cameraId, version })
      await loadBaselineVersions(cameraId)
      await loadBaselines()
    } finally {
      activatingVersion.value = null
    }
  }

  async function retireBaselineVersion(cameraId: string, version: string, reason: string) {
    await apiRetireBaseline({ camera_id: cameraId, version, reason })
    await loadBaselineVersions(cameraId)
    await loadBaselines()
  }

  async function deleteBaselineVersionAction(cameraId: string, version: string) {
    deletingBaseline.value = `${cameraId}-${version}`
    try {
      await apiDeleteBaselineVersion({ camera_id: cameraId, version })
      if (versionDrawerVisible.value && versionDrawerCamera.value === cameraId) {
        await loadBaselineVersions(cameraId)
      }
      await loadBaselines()
    } finally {
      deletingBaseline.value = null
    }
  }

  // ── Group / false-positive merge actions ──
  async function mergeGroupBaselineAction(groupId: string) {
    mergingGroup.value = groupId
    try {
      const res = await apiMergeGroupBaseline({ group_id: groupId })
      await loadCameraGroups()
      return res
    } finally {
      mergingGroup.value = null
    }
  }

  async function mergeFalsePositivesAction(cameraId: string) {
    mergingFP.value = cameraId
    try {
      const res = await apiMergeFalsePositives({ camera_id: cameraId })
      await loadBaselines()
      return res
    } finally {
      mergingFP.value = null
    }
  }

  // ── Optimize (two-step: preview then execute) ──
  async function previewOptimize(cameraId: string, zoneId = 'default', targetRatio = 0.2) {
    return apiPreviewOptimize({ camera_id: cameraId, zone_id: zoneId, target_ratio: targetRatio })
  }

  async function executeOptimize(cameraId: string, version: string, zoneId = 'default', targetRatio = 0.2) {
    optimizingBaseline.value = `${cameraId}-${version}`
    try {
      const res = await apiOptimizeBaseline({ camera_id: cameraId, zone_id: zoneId, target_ratio: targetRatio })
      await loadBaselines()
      return res
    } finally {
      optimizingBaseline.value = null
    }
  }

  function setOptimizing(key: string | null) {
    optimizingBaseline.value = key
  }

  return {
    baselines,
    baselinesLoading,
    baselineVersions,
    versionsLoading,
    versionDrawerVisible,
    versionDrawerCamera,
    cameraGroups,
    groupsLoading,
    verifyingVersion,
    activatingVersion,
    deletingBaseline,
    mergingGroup,
    mergingFP,
    optimizingBaseline,
    loadBaselines,
    loadBaselineVersions,
    loadCameraGroups,
    openVersionDrawer,
    closeVersionDrawer,
    verifyBaselineVersion,
    activateBaselineVersion,
    retireBaselineVersion,
    deleteBaselineVersion: deleteBaselineVersionAction,
    mergeGroupBaseline: mergeGroupBaselineAction,
    mergeFalsePositives: mergeFalsePositivesAction,
    previewOptimize,
    executeOptimize,
    setOptimizing,
  }
})
