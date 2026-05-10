<template>
  <div class="warehouse-page">
    <div class="top-bar">
      <div>
        <div class="page-title">三维仓库监控点位</div>
        <div class="page-subtitle">Warehouse 3D Monitoring Visualization</div>
      </div>

      <div class="stat-group">
        <a-card size="small" class="stat-card">
          <div class="stat-value">{{ cameraList.length }}</div>
          <div class="stat-label">摄像头总数</div>
        </a-card>
        <a-card size="small" class="stat-card online">
          <div class="stat-value">{{ onlineCount }}</div>
          <div class="stat-label">在线</div>
        </a-card>
        <a-card size="small" class="stat-card offline">
          <div class="stat-value">{{ offlineCount }}</div>
          <div class="stat-label">离线</div>
        </a-card>
        <a-card size="small" class="stat-card alarm">
          <div class="stat-value">{{ alarmCount }}</div>
          <div class="stat-label">告警</div>
        </a-card>
      </div>
    </div>

    <div class="main-layout">
      <div class="left-panel">
        <a-card title="监控点位列表" class="panel-card">
          <a-input-search
            v-model:value="keyword"
            placeholder="搜索摄像头名称、区域或 IP"
            allow-clear
            style="margin-bottom: 12px"
          />

          <a-list
            :data-source="filteredCameraList"
            item-layout="horizontal"
            class="camera-list"
          >
            <template #renderItem="{ item }">
              <a-list-item
                class="camera-item"
                :class="{ active: selectedCamera?.id === item.id }"
                @click="selectCamera(item)"
              >
                <a-list-item-meta>
                  <template #avatar>
                    <div class="camera-dot" :class="item.status"></div>
                  </template>

                  <template #title>
                    <div class="camera-name">{{ item.name }}</div>
                  </template>

                  <template #description>
                    <div class="camera-desc">
                      {{ item.area }} / {{ item.ip }}
                    </div>
                  </template>
                </a-list-item-meta>

                <a-tag :color="getStatusColor(item.status)">
                  {{ getStatusText(item.status) }}
                </a-tag>
              </a-list-item>
            </template>
          </a-list>
        </a-card>
      </div>

      <div class="center-panel">
        <div ref="threeContainer" class="three-container"></div>

        <div class="scene-tools">
          <a-button size="small" @click="resetCameraView">重置视角</a-button>
          <a-button size="small" @click="focusPlanView">平面俯览</a-button>
          <a-button size="small" @click="toggleCoverage">
            {{ showCoverage ? '隐藏覆盖范围' : '显示覆盖范围' }}
          </a-button>
        </div>

        <div class="scene-legend">
          <span><i class="legend-dot online"></i> 在线点位</span>
          <span><i class="legend-dot offline"></i> 离线点位</span>
          <span><i class="legend-dot alarm"></i> 告警点位</span>
          <span><i class="legend-swatch coverage"></i> 监控覆盖</span>
        </div>
      </div>

      <div class="right-panel">
        <a-card title="点位详情" class="panel-card">
          <template v-if="selectedCamera">
            <a-descriptions bordered size="small" :column="1">
              <a-descriptions-item label="摄像头名称">
                {{ selectedCamera.name }}
              </a-descriptions-item>

              <a-descriptions-item label="设备状态">
                <a-tag :color="getStatusColor(selectedCamera.status)">
                  {{ getStatusText(selectedCamera.status) }}
                </a-tag>
              </a-descriptions-item>

              <a-descriptions-item label="安装区域">
                {{ selectedCamera.area }}
              </a-descriptions-item>

              <a-descriptions-item label="IP 地址">
                {{ selectedCamera.ip }}
              </a-descriptions-item>

              <a-descriptions-item label="安装高度">
                {{ selectedCamera.height }} 米
              </a-descriptions-item>

              <a-descriptions-item label="监控朝向">
                {{ selectedCamera.directionText }}
              </a-descriptions-item>

              <a-descriptions-item label="覆盖范围">
                {{ selectedCamera.coverage }}
              </a-descriptions-item>
            </a-descriptions>

            <div class="video-box">
              <div class="video-placeholder">
                <div class="video-title">{{ selectedCamera.name }}</div>
                <div class="video-text">此处预留实时视频画面</div>
                <div class="video-url">RTSP / WebRTC / FLV / HLS</div>
              </div>
            </div>

            <div class="action-group">
              <a-button type="primary" block>查看实时视频</a-button>
              <a-button block>查看历史录像</a-button>
              <a-button danger block v-if="selectedCamera.status === 'alarm'">
                处理告警
              </a-button>
            </div>
          </template>

          <a-empty v-else description="请选择一个摄像头点位" />
        </a-card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import monitoringIconUrl from '../assets/monitoring.png'

const warehouseSize = {
  width: 182,
  depth: 116,
  wallHeight: 14
}

const palette = {
  floor: 0x10243a,
  floorLine: 0x2dd7ff,
  wall: 0x1fbce8,
  aisle: 0xb8c0c6,
  room: 0xf4ce38,
  rackBlue: 0x20a7e8,
  rackPanel: 0xaec2d2,
  rackFrame: 0x6b8394,
  glow: 0x18d9ff
}

const roomBlocks = [
  { name: '第三调试间', x: -55, z: 4, w: 42, d: 17, h: 3.8 },
  { name: '第二调试间', x: -55, z: -17, w: 42, d: 17, h: 3.8 },
  { name: '第一调试间', x: -55, z: -38, w: 42, d: 18, h: 3.8 },
  { name: '仓库办公室', x: -12, z: -40, w: 31, d: 22, h: 3.8 }
]

const aisleBlocks = [
  { x: -28, z: -16, w: 4.6, d: 92 },
  { x: -5, z: -50, w: 52, d: 4.6 }
]

const shelfBlocks = [
  { name: '三层货架', x: -18, z: 37, w: 11, d: 31, h: 9.2, levels: 3, verticalText: true, color: palette.rackBlue },
  { name: '三层货架', x: -18, z: -9, w: 11, d: 39, h: 9.2, levels: 3, verticalText: true, color: palette.rackBlue },
  { name: '三层货架', x: 38, z: 10, w: 11, d: 53, h: 9.2, levels: 3, verticalText: true, color: palette.rackBlue },
  { name: '三层货架', x: 51, z: 10, w: 11, d: 53, h: 9.2, levels: 3, verticalText: true, color: palette.rackBlue },
  { name: '三层小货架', x: 72, z: -48, w: 54, d: 8, h: 7.2, levels: 3, color: palette.rackBlue },
  { name: '', x: 42, z: 50, w: 112, d: 8, h: 8.2, levels: 3, color: palette.rackBlue }
]

const doorLabels = [
  { name: '入口', x: -80, z: -55, w: 16, d: 5.5, rotate: 0 },
  { name: '仓库大门', x: 91, z: -10, w: 11, d: 23, rotate: Math.PI / 2 }
]

const cameraList = ref([
  {
    id: 1,
    name: '入口监控',
    area: '入口',
    ip: '192.168.1.101',
    status: 'online',
    x: -88,
    y: 6.5,
    z: -55,
    direction: 68,
    directionText: '朝向入口与第一调试间外侧通道',
    height: 3.5,
    coverage: '入口、第一调试间外侧通道',
    range: 30,
    fov: 64
  },
  {
    id: 2,
    name: '左上角监控',
    area: '西北空区',
    ip: '192.168.1.102',
    status: 'online',
    x: -84,
    y: 7.2,
    z: 50,
    direction: 126,
    directionText: '朝向北侧通道与左侧空区',
    height: 3.8,
    coverage: '左上角空区、北侧通道',
    range: 30,
    fov: 66
  },
  {
    id: 3,
    name: '北侧主通道监控',
    area: '北侧主通道',
    ip: '192.168.1.103',
    status: 'online',
    x: -28,
    y: 7.5,
    z: 50,
    direction: 220,
    directionText: '朝向灰色主通道',
    height: 4,
    coverage: '主通道、上侧三层货架',
    range: 28,
    fov: 70
  },
  {
    id: 4,
    name: '货架入口监控',
    area: '上侧三层货架',
    ip: '192.168.1.104',
    status: 'online',
    x: -9,
    y: 8,
    z: 50,
    direction: 150,
    directionText: '朝向三层货架通道',
    height: 4,
    coverage: '上侧三层货架与中部货架口',
    range: 20,
    fov: 68
  },
  {
    id: 5,
    name: '右上货架监控',
    area: '北侧长货架',
    ip: '192.168.1.105',
    status: 'online',
    x: 84,
    y: 7.2,
    z: 50,
    direction: 224,
    directionText: '朝向北侧长货架',
    height: 3.8,
    coverage: '北侧长货架右端、右侧货架区',
    range: 31,
    fov: 66
  },
  {
    id: 6,
    name: '办公室外监控',
    area: '仓库办公室',
    ip: '192.168.1.106',
    status: 'online',
    x: 15,
    y: 6.8,
    z: -49,
    direction: 332,
    directionText: '朝向办公室与南侧通道',
    height: 3.5,
    coverage: '仓库办公室、三层小货架西侧',
    range: 28,
    fov: 64
  }
])

const keyword = ref('')
const selectedCamera = ref(null)
const threeContainer = ref(null)
const showCoverage = ref(true)

const filteredCameraList = computed(() => {
  const value = keyword.value.trim()
  if (!value) return cameraList.value

  return cameraList.value.filter((item) => {
    return item.name.includes(value) ||
      item.area.includes(value) ||
      item.ip.includes(value)
  })
})

const onlineCount = computed(() => cameraList.value.filter((item) => item.status === 'online').length)
const offlineCount = computed(() => cameraList.value.filter((item) => item.status === 'offline').length)
const alarmCount = computed(() => cameraList.value.filter((item) => item.status === 'alarm').length)

let scene = null
let renderer = null
let camera = null
let controls = null
let raycaster = null
let mouse = null
let animationId = null

const cameraMeshMap = new Map()
const coverageMeshList = []
const disposableTextures = []
const disposableMaterials = []
const disposableGeometries = []
const monitoringTexture = new THREE.TextureLoader().load(monitoringIconUrl)
monitoringTexture.colorSpace = THREE.SRGBColorSpace
disposableTextures.push(monitoringTexture)

function getStatusText(status) {
  const map = {
    online: '在线',
    offline: '离线',
    alarm: '告警'
  }
  return map[status] || '未知'
}

function getStatusColor(status) {
  const map = {
    online: 'green',
    offline: 'default',
    alarm: 'red'
  }
  return map[status] || 'default'
}

function getThreeColor(status) {
  const map = {
    online: 0x55f75a,
    offline: 0x9aa8b5,
    alarm: 0xff4d4f
  }
  return map[status] || 0x55f75a
}

function initThree() {
  const { clientWidth: width, clientHeight: height } = threeContainer.value

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x061525)
  scene.fog = new THREE.Fog(0x061525, 132, 260)

  camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1000)
  resetCameraPosition()

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
  })
  renderer.setSize(width, height)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.1
  threeContainer.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.maxPolarAngle = Math.PI / 2.08
  controls.minDistance = 42
  controls.maxDistance = 220
  controls.target.set(0, 0, 0)

  raycaster = new THREE.Raycaster()
  mouse = new THREE.Vector2()

  addLights()
  addWarehouse()
  addCameraPoints()

  renderer.domElement.addEventListener('click', onSceneClick)
  window.addEventListener('resize', onWindowResize)

  animate()
}

function resetCameraPosition() {
  camera.position.set(42, 104, 112)
  camera.lookAt(0, 0, 0)
}

function addLights() {
  scene.add(new THREE.AmbientLight(0xb8dfff, 0.62))

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.45)
  dirLight.position.set(36, 92, 58)
  dirLight.castShadow = true
  dirLight.shadow.mapSize.width = 2048
  dirLight.shadow.mapSize.height = 2048
  dirLight.shadow.camera.near = 10
  dirLight.shadow.camera.far = 220
  dirLight.shadow.camera.left = -120
  dirLight.shadow.camera.right = 120
  dirLight.shadow.camera.top = 120
  dirLight.shadow.camera.bottom = -120
  scene.add(dirLight)

  const cyanLight = new THREE.PointLight(palette.glow, 2.1, 190)
  cyanLight.position.set(4, 36, 10)
  scene.add(cyanLight)
}

function addWarehouse() {
  addOuterGrid()
  addFloor()
  addWalls()
  addAisles()
  addRooms()
  addShelves()
  addDoorLabels()
}

function addOuterGrid() {
  const outerGrid = new THREE.GridHelper(250, 50, 0x0f7194, 0x0a3552)
  outerGrid.position.y = -0.82
  outerGrid.material.transparent = true
  outerGrid.material.opacity = 0.38
  scene.add(outerGrid)

  const halo = addBox(warehouseSize.width + 14, 0.12, warehouseSize.depth + 14, 0, -0.76, 0, createMaterial({
    color: 0x0b4f78,
    transparent: true,
    opacity: 0.16,
    roughness: 0.62
  }))
  halo.receiveShadow = true
}

function addFloor() {
  const floorMaterial = createMaterial({
    color: palette.floor,
    roughness: 0.54,
    metalness: 0.12
  })
  const floor = addBox(warehouseSize.width, 0.9, warehouseSize.depth, 0, -0.45, 0, floorMaterial)
  floor.receiveShadow = true

  const grid = new THREE.GridHelper(warehouseSize.width, 26, 0x27d8ff, 0x24506a)
  grid.position.y = 0.04
  grid.material.transparent = true
  grid.material.opacity = 0.46
  scene.add(grid)

  addFloorBorder()
}

function addFloorBorder() {
  const points = [
    [-warehouseSize.width / 2, 0.18, -warehouseSize.depth / 2],
    [warehouseSize.width / 2, 0.18, -warehouseSize.depth / 2],
    [warehouseSize.width / 2, 0.18, warehouseSize.depth / 2],
    [-warehouseSize.width / 2, 0.18, warehouseSize.depth / 2],
    [-warehouseSize.width / 2, 0.18, -warehouseSize.depth / 2]
  ].map(([x, y, z]) => new THREE.Vector3(x, y, z))

  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: palette.glow,
    transparent: true,
    opacity: 0.9
  })
  trackDisposable(geometry, material)

  const line = new THREE.Line(geometry, material)
  scene.add(line)
}

function addWalls() {
  const wallMaterial = createMaterial({
    color: palette.wall,
    transparent: true,
    opacity: 0.08,
    roughness: 0.36,
    metalness: 0.08,
    emissive: 0x075f84,
    emissiveIntensity: 0.34
  })
  const capMaterial = createMaterial({
    color: 0x18d6ff,
    transparent: true,
    opacity: 0.18,
    emissive: 0x18d6ff,
    emissiveIntensity: 0.55
  })

  const halfW = warehouseSize.width / 2
  const halfD = warehouseSize.depth / 2
  const wallH = warehouseSize.wallHeight
  const wallY = wallH / 2

  addWallSegment(warehouseSize.width, wallH, 1.1, 0, wallY, halfD, wallMaterial, capMaterial)
  addWallSegment(warehouseSize.width, wallH, 1.1, 0, wallY, -halfD, wallMaterial, capMaterial)
  addWallSegment(1.1, wallH, warehouseSize.depth, -halfW, wallY, 0, wallMaterial, capMaterial)
  addWallSegment(1.1, wallH, 35, halfW, wallY, 40.5, wallMaterial, capMaterial)
  addWallSegment(1.1, wallH, 34, halfW, wallY, -41, wallMaterial, capMaterial)
}

function addWallSegment(w, h, d, x, y, z, material, capMaterial) {
  const wall = addBox(w, h, d, x, y, z, material)
  wall.receiveShadow = true

  const cap = addBox(w + (w > d ? 0 : 0.8), 0.34, d + (d > w ? 0 : 0.8), x, h + 0.22, z, capMaterial)
  cap.castShadow = false

  addEdges(wall, 0x52e8ff, 0.68)
}

function addAisles() {
  const material = createMaterial({
    color: palette.aisle,
    transparent: true,
    opacity: 0.28,
    roughness: 0.72,
    metalness: 0.03
  })

  aisleBlocks.forEach((aisle) => {
    const mesh = addBox(aisle.w, 0.24, aisle.d, aisle.x, 0.16, aisle.z, material)
    mesh.receiveShadow = true
    addEdges(mesh, 0xd7dde2, 0.76)
  })
}

function addRooms() {
  roomBlocks.forEach((room) => {
    const material = createMaterial({
      color: palette.room,
      transparent: true,
      opacity: 0.14,
      roughness: 0.5,
      metalness: 0.04,
      emissive: 0x604600,
      emissiveIntensity: 0.05
    })
    const mesh = addBox(room.w, room.h, room.d, room.x, room.h / 2, room.z, material)
    mesh.castShadow = true
    mesh.receiveShadow = true

    addEdges(mesh, 0xf4ce38, 0.92)
    addTopLabel(room.name, room.x, room.h + 0.08, room.z, room.w * 0.78, room.d * 0.68, {
      fontSize: 36,
      color: '#15120a'
    })
  })
}

function addShelves() {
  shelfBlocks.forEach((shelf) => {
    const shelfGroup = createShelfModel(shelf)
    shelfGroup.position.set(shelf.x, 0, shelf.z)
    scene.add(shelfGroup)

    if (shelf.name) {
      addTopLabel(shelf.name, shelf.x, shelf.h + 0.14, shelf.z, shelf.w * 0.9, shelf.d * 0.84, {
        vertical: shelf.verticalText,
        fontSize: shelf.verticalText ? 52 : 38,
        color: '#04141f'
      })
    }
  })
}

function addDoorLabels() {
  doorLabels.forEach((label) => {
    const plate = addBox(label.w, 0.2, label.d, label.x, 0.22, label.z, createMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.1,
      roughness: 0.46
    }))
    plate.rotation.y = label.rotate
    addEdges(plate, 0x0e1418, 1)

    addTopLabel(label.name, label.x, 0.4, label.z, label.w * 0.88, label.d * 0.58, {
      fontSize: 34,
      color: '#111111',
      rotate: label.rotate
    })
  })
}

function createShelfModel(shelf) {
  const group = new THREE.Group()
  const fill = new THREE.Mesh(
    new THREE.BoxGeometry(shelf.w, shelf.h, shelf.d),
    createMaterial({
      color: shelf.color,
      transparent: true,
      opacity: 0.08,
      roughness: 0.45,
      metalness: 0.1,
      emissive: 0x064c6f,
      emissiveIntensity: 0.08
    })
  )
  trackDisposable(fill.geometry)
  fill.position.y = shelf.h / 2
  fill.receiveShadow = true
  group.add(fill)

  group.add(createOutlineBox(shelf.w, shelf.h, shelf.d, shelf.color, 0.95))

  for (let index = 1; index < shelf.levels; index += 1) {
    group.add(createRectangleLine(shelf.w, shelf.d, shelf.h / shelf.levels * index, shelf.color, 0.72))
  }

  return group
}

function addShelfGoods(group, shelf) {
  const boxMaterial = createMaterial({
    color: 0xd5e0e8,
    roughness: 0.62,
    metalness: 0.02
  })
  const accentMaterial = createMaterial({
    color: 0x89a9be,
    roughness: 0.58
  })
  const lanes = Math.max(4, Math.floor(shelf.d / 9))

  for (let level = 0; level < shelf.levels; level += 1) {
    const y = (shelf.h / shelf.levels) * level + 0.86
    for (let index = 0; index < lanes; index += 1) {
      if ((index + level) % 4 === 0) continue

      const cargo = new THREE.Mesh(
        new THREE.BoxGeometry(shelf.w * 0.68, 0.9, 3.4),
        (index + level) % 3 === 0 ? accentMaterial : boxMaterial
      )
      trackDisposable(cargo.geometry)
      cargo.position.set(0, y, -shelf.d / 2 + 4.5 + index * (shelf.d - 9) / Math.max(1, lanes - 1))
      cargo.castShadow = true
      cargo.receiveShadow = true
      group.add(cargo)
    }
  }
}

function addCameraPoints() {
  cameraList.value.forEach((item) => {
    const group = new THREE.Group()
    group.position.set(item.x, item.y, item.z)
    group.userData = {
      type: 'camera',
      id: item.id
    }

    const bodyColor = getThreeColor(item.status)
    const iconMaterial = new THREE.SpriteMaterial({
      map: monitoringTexture,
      transparent: true,
      depthWrite: false
    })
    trackDisposable(iconMaterial)

    const icon = new THREE.Sprite(iconMaterial)
    icon.scale.set(7.2, 7.2, 1)
    group.add(icon)
    group.userData.icon = icon

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.55, 0.08, 10, 36),
      createMaterial({
        color: bodyColor,
        emissive: bodyColor,
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0.86
      })
    )
    trackDisposable(ring.geometry)
    ring.rotation.x = Math.PI / 2
    group.add(ring)
    group.userData.ring = ring

    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.14, 0.14, item.y, 12),
      createMaterial({ color: 0xe9f5ff, roughness: 0.45, metalness: 0.14 })
    )
    trackDisposable(pole.geometry)
    pole.position.y = -item.y / 2
    group.add(pole)

    const arrow = new THREE.ArrowHelper(
      directionToVector(item.direction),
      new THREE.Vector3(0, 0, 0),
      8.5,
      bodyColor,
      2.1,
      1.15
    )
    group.add(arrow)

    const coverage = createCoverageMesh(item)
    coverage.visible = showCoverage.value
    scene.add(coverage)
    coverageMeshList.push(coverage)

    scene.add(group)
    cameraMeshMap.set(item.id, group)

    createTextSprite(item.name, item.x, item.y + 3.4, item.z)
  })
}

function createCoverageMesh(item) {
  const angle = item.fov || 70
  const segments = 36
  const positions = [0, 0, 0]
  const indices = []
  const edgePoints = [new THREE.Vector3(0, 0, 0)]

  for (let index = 0; index <= segments; index += 1) {
    const offsetAngle = -angle / 2 + (angle / segments) * index
    const direction = directionToVector(item.direction + offsetAngle)
    const distance = getIndoorCoverageDistance(item, direction)
    const x = direction.x * distance
    const z = direction.z * distance

    positions.push(x, 0, z)
    edgePoints.push(new THREE.Vector3(x, 0, z))

    if (index < segments) {
      indices.push(0, index + 1, index + 2)
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  const material = new THREE.MeshBasicMaterial({
    color: getThreeColor(item.status),
    transparent: true,
    opacity: item.status === 'alarm' ? 0.27 : 0.18,
    side: THREE.DoubleSide,
    depthWrite: false
  })
  trackDisposable(geometry, material)

  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(item.x, 0.12, item.z)

  edgePoints.push(new THREE.Vector3(0, 0, 0))
  const edgeGeometry = new THREE.BufferGeometry().setFromPoints(edgePoints)
  const edge = new THREE.Line(
    edgeGeometry,
    new THREE.LineBasicMaterial({
      color: getThreeColor(item.status),
      transparent: true,
      opacity: 0.42
    })
  )
  trackDisposable(edge.geometry, edge.material)
  edge.position.set(item.x, 0.16, item.z)
  edge.visible = showCoverage.value
  scene.add(edge)
  coverageMeshList.push(edge)

  return mesh
}

function getIndoorCoverageDistance(item, direction) {
  const padding = 2.4
  const minX = -warehouseSize.width / 2 + padding
  const maxX = warehouseSize.width / 2 - padding
  const minZ = -warehouseSize.depth / 2 + padding
  const maxZ = warehouseSize.depth / 2 - padding
  const candidates = [item.range || 24]

  if (direction.x > 0) {
    candidates.push((maxX - item.x) / direction.x)
  } else if (direction.x < 0) {
    candidates.push((minX - item.x) / direction.x)
  }

  if (direction.z > 0) {
    candidates.push((maxZ - item.z) / direction.z)
  } else if (direction.z < 0) {
    candidates.push((minZ - item.z) / direction.z)
  }

  const distance = Math.min(...candidates.filter((value) => value > 0))
  return Math.max(0.5, distance)
}

function addTopLabel(text, x, y, z, w, d, options = {}) {
  const texture = createLabelTexture(text, options)
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide
  })
  trackDisposable(null, material)

  const geometry = new THREE.PlaneGeometry(w, d)
  trackDisposable(geometry)
  const plane = new THREE.Mesh(geometry, material)
  plane.rotation.x = -Math.PI / 2
  plane.rotation.z = -(options.rotate || 0)
  plane.position.set(x, y, z)
  scene.add(plane)
}

function createLabelTexture(text, options = {}) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  canvas.width = options.vertical ? 256 : 512
  canvas.height = options.vertical ? 512 : 192

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = options.color || '#111111'
  ctx.font = `700 ${options.fontSize || 34}px "Microsoft YaHei", "PingFang SC", sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  if (options.vertical) {
    const chars = Array.from(text)
    const lineHeight = canvas.height / (chars.length + 1)
    chars.forEach((char, index) => {
      ctx.fillText(char, canvas.width / 2, lineHeight * (index + 1))
    })
  } else {
    ctx.fillText(text, canvas.width / 2, canvas.height / 2)
  }

  const texture = new THREE.CanvasTexture(canvas)
  disposableTextures.push(texture)
  return texture
}

function createTextSprite(text, x, y, z) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  canvas.width = 320
  canvas.height = 88

  const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
  gradient.addColorStop(0, 'rgba(7, 22, 38, 0.86)')
  gradient.addColorStop(1, 'rgba(10, 52, 75, 0.82)')
  ctx.fillStyle = gradient
  roundRect(ctx, 0, 0, canvas.width, canvas.height, 18)
  ctx.fill()

  ctx.strokeStyle = 'rgba(86, 232, 255, 0.78)'
  ctx.lineWidth = 3
  roundRect(ctx, 3, 3, canvas.width - 6, canvas.height - 6, 16)
  ctx.stroke()

  ctx.fillStyle = '#ffffff'
  ctx.font = '700 28px "Microsoft YaHei", "PingFang SC", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, canvas.width / 2, canvas.height / 2)

  const texture = new THREE.CanvasTexture(canvas)
  disposableTextures.push(texture)

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true
  })
  trackDisposable(null, material)

  const sprite = new THREE.Sprite(material)
  sprite.position.set(x, y, z)
  sprite.scale.set(13, 3.6, 1)

  scene.add(sprite)
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

function addBox(w, h, d, x, y, z, material) {
  const geometry = new THREE.BoxGeometry(w, h, d)
  trackDisposable(geometry)
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(x, y, z)
  scene.add(mesh)
  return mesh
}

function addEdges(mesh, color, opacity) {
  const geometry = new THREE.EdgesGeometry(mesh.geometry)
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity })
  trackDisposable(geometry, material)
  const edge = new THREE.LineSegments(geometry, material)
  edge.position.copy(mesh.position)
  edge.rotation.copy(mesh.rotation)
  scene.add(edge)
}

function createOutlineBox(w, h, d, color, opacity = 0.86) {
  const geometry = new THREE.BoxGeometry(w, h, d)
  const edgeGeometry = new THREE.EdgesGeometry(geometry)
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity
  })
  trackDisposable(geometry, edgeGeometry, material)

  const line = new THREE.LineSegments(edgeGeometry, material)
  line.position.y = h / 2
  return line
}

function createRectangleLine(w, d, y, color, opacity = 0.72) {
  const points = [
    new THREE.Vector3(-w / 2, y, -d / 2),
    new THREE.Vector3(w / 2, y, -d / 2),
    new THREE.Vector3(w / 2, y, d / 2),
    new THREE.Vector3(-w / 2, y, d / 2),
    new THREE.Vector3(-w / 2, y, -d / 2)
  ]
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity
  })
  trackDisposable(geometry, material)

  return new THREE.Line(geometry, material)
}

function createMaterial(options) {
  const material = new THREE.MeshStandardMaterial(options)
  disposableMaterials.push(material)
  return material
}

function trackDisposable(...items) {
  items.forEach((item) => {
    if (!item) return

    if (item.isMaterial) {
      disposableMaterials.push(item)
    } else if (typeof item.dispose === 'function') {
      disposableGeometries.push(item)
    }
  })
}

function directionToVector(direction) {
  const rad = THREE.MathUtils.degToRad(direction)
  return new THREE.Vector3(
    Math.sin(rad),
    0,
    Math.cos(rad)
  ).normalize()
}

function onSceneClick(event) {
  const rect = renderer.domElement.getBoundingClientRect()

  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  raycaster.setFromCamera(mouse, camera)

  const objects = Array.from(cameraMeshMap.values())
  const intersects = raycaster.intersectObjects(objects, true)

  if (intersects.length > 0) {
    let object = intersects[0].object

    while (object.parent && !object.userData?.type) {
      object = object.parent
    }

    if (object.userData?.type === 'camera') {
      const id = object.userData.id
      const target = cameraList.value.find((item) => item.id === id)
      if (target) selectCamera(target)
    }
  }
}

function selectCamera(item) {
  selectedCamera.value = item

  const mesh = cameraMeshMap.get(item.id)
  if (!mesh) return

  controls.target.set(item.x, item.y, item.z)
  camera.position.set(item.x + 24, item.y + 20, item.z + 26)
  controls.update()

  highlightCamera(item.id)
}

function highlightCamera(id) {
  cameraMeshMap.forEach((group, key) => {
    const icon = group.userData.icon
    const ring = group.userData.ring
    if (!icon || !ring) return

    if (key === id) {
      icon.scale.set(9, 9, 1)
      ring.scale.set(1.32, 1.32, 1.32)
    } else {
      icon.scale.set(7.2, 7.2, 1)
      ring.scale.set(1, 1, 1)
    }
  })
}

function resetCameraView() {
  resetCameraPosition()
  controls.target.set(0, 0, 0)
  controls.update()
}

function focusPlanView() {
  camera.position.set(0, 150, 0.01)
  controls.target.set(0, 0, 0)
  controls.update()
}

function toggleCoverage() {
  showCoverage.value = !showCoverage.value
  coverageMeshList.forEach((mesh) => {
    mesh.visible = showCoverage.value
  })
}

function onWindowResize() {
  if (!threeContainer.value || !camera || !renderer) return

  const width = threeContainer.value.clientWidth
  const height = threeContainer.value.clientHeight

  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

function animate() {
  animationId = requestAnimationFrame(animate)

  controls.update()

  const time = Date.now() * 0.004
  cameraMeshMap.forEach((group, id) => {
    const item = cameraList.value.find((cameraItem) => cameraItem.id === id)
    const icon = group.userData.icon
    const ring = group.userData.ring
    if (!item || !icon || !ring) return

    const pulse = 1 + Math.sin(time + id) * 0.08
    ring.scale.set(pulse, pulse, pulse)

    if (item.status === 'alarm') {
      const alarmScale = 1 + Math.sin(time * 1.35) * 0.18
      icon.scale.set(7.2 * alarmScale, 7.2 * alarmScale, 1)
    }
  })

  renderer.render(scene, camera)
}

watch(showCoverage, (val) => {
  coverageMeshList.forEach((mesh) => {
    mesh.visible = val
  })
})

onMounted(() => {
  initThree()
  selectedCamera.value = cameraList.value[0]
  highlightCamera(cameraList.value[0].id)
})

onBeforeUnmount(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }

  if (renderer?.domElement) {
    renderer.domElement.removeEventListener('click', onSceneClick)
  }

  window.removeEventListener('resize', onWindowResize)
  disposableTextures.forEach((texture) => texture.dispose())
  disposableMaterials.forEach((material) => material.dispose())
  disposableGeometries.forEach((geometry) => geometry.dispose())

  if (renderer) {
    renderer.dispose()
  }
})
</script>

<style scoped>
.warehouse-page {
  width: 100%;
  height: 100vh;
  background:
    radial-gradient(circle at 48% 22%, rgba(28, 203, 255, 0.18), transparent 28%),
    linear-gradient(135deg, #06101d 0%, #071827 48%, #030b13 100%);
  color: #fff;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.top-bar {
  height: 86px;
  padding: 16px 20px;
  background:
    linear-gradient(90deg, rgba(8, 24, 43, 0.95), rgba(13, 53, 80, 0.86)),
    repeating-linear-gradient(90deg, transparent 0, transparent 24px, rgba(66, 208, 255, 0.05) 25px);
  border-bottom: 1px solid rgba(80, 226, 255, 0.26);
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 16px 38px rgba(0, 0, 0, 0.22);
}

.page-title {
  color: #fff;
  font-size: 24px;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.page-subtitle {
  color: rgba(189, 236, 255, 0.66);
  margin-top: 4px;
  font-size: 13px;
  letter-spacing: 0.04em;
}

.stat-group {
  display: flex;
  gap: 12px;
}

.stat-card {
  width: 120px;
  background: rgba(255, 255, 255, 0.055);
  border: 1px solid rgba(92, 224, 255, 0.2);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.stat-value {
  color: #fff;
  font-size: 24px;
  font-weight: 800;
  line-height: 1;
}

.stat-label {
  color: rgba(213, 241, 255, 0.64);
  margin-top: 6px;
}

.stat-card.online .stat-value {
  color: #55f75a;
}

.stat-card.offline .stat-value {
  color: #9aa8b5;
}

.stat-card.alarm .stat-value {
  color: #ff5a61;
}

.main-layout {
  flex: 1;
  display: grid;
  grid-template-columns: 310px 1fr 360px;
  gap: 14px;
  padding: 14px;
  min-height: 0;
}

.left-panel,
.right-panel,
.center-panel {
  min-height: 0;
}

.panel-card {
  height: 100%;
  background:
    linear-gradient(180deg, rgba(13, 35, 57, 0.96), rgba(7, 22, 36, 0.94));
  border: 1px solid rgba(86, 232, 255, 0.2);
  color: #fff;
}

.camera-list {
  height: calc(100vh - 210px);
  overflow-y: auto;
}

.camera-item {
  cursor: pointer;
  padding: 12px !important;
  border-radius: 10px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid transparent;
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
}

.camera-item:hover,
.camera-item.active {
  background: rgba(34, 193, 255, 0.15);
  border-color: rgba(86, 232, 255, 0.52);
  transform: translateX(2px);
}

.camera-name {
  color: #fff;
  font-weight: 700;
}

.camera-desc {
  color: rgba(213, 241, 255, 0.58);
  font-size: 12px;
}

.camera-dot,
.legend-dot {
  display: inline-block;
  border-radius: 50%;
}

.camera-dot {
  width: 12px;
  height: 12px;
  margin-top: 6px;
}

.camera-dot.online,
.legend-dot.online {
  background: #55f75a;
  box-shadow: 0 0 10px rgba(85, 247, 90, 0.58);
}

.camera-dot.offline,
.legend-dot.offline {
  background: #9aa8b5;
}

.camera-dot.alarm,
.legend-dot.alarm {
  background: #ff4d4f;
  box-shadow: 0 0 12px rgba(255, 77, 79, 0.85);
}

.center-panel {
  position: relative;
  background:
    radial-gradient(circle at center, rgba(13, 74, 105, 0.45) 0%, rgba(4, 13, 24, 0.98) 72%),
    repeating-linear-gradient(0deg, rgba(64, 211, 255, 0.035) 0, rgba(64, 211, 255, 0.035) 1px, transparent 1px, transparent 22px),
    repeating-linear-gradient(90deg, rgba(64, 211, 255, 0.035) 0, rgba(64, 211, 255, 0.035) 1px, transparent 1px, transparent 22px);
  border: 1px solid rgba(86, 232, 255, 0.22);
  overflow: hidden;
  border-radius: 14px;
  box-shadow: inset 0 0 42px rgba(24, 217, 255, 0.08);
}

.three-container {
  width: 100%;
  height: 100%;
}

.scene-tools {
  position: absolute;
  left: 16px;
  top: 16px;
  display: flex;
  gap: 8px;
  padding: 8px;
  border-radius: 12px;
  background: rgba(4, 14, 25, 0.64);
  border: 1px solid rgba(86, 232, 255, 0.2);
  backdrop-filter: blur(8px);
}

.scene-legend {
  position: absolute;
  left: 16px;
  bottom: 16px;
  padding: 9px 12px;
  border-radius: 10px;
  background: rgba(4, 14, 25, 0.72);
  border: 1px solid rgba(86, 232, 255, 0.22);
  display: flex;
  gap: 16px;
  color: rgba(222, 245, 255, 0.82);
  backdrop-filter: blur(8px);
}

.legend-dot {
  width: 9px;
  height: 9px;
  margin-right: 6px;
}

.legend-swatch {
  display: inline-block;
  width: 15px;
  height: 9px;
  margin-right: 6px;
  border-radius: 999px;
  vertical-align: middle;
}

.legend-swatch.coverage {
  background: rgba(85, 247, 90, 0.28);
  border: 1px solid rgba(85, 247, 90, 0.5);
}

.video-box {
  margin-top: 16px;
  height: 170px;
  border-radius: 10px;
  background: #000;
  border: 1px solid rgba(86, 232, 255, 0.24);
  overflow: hidden;
}

.video-placeholder {
  height: 100%;
  background:
    linear-gradient(45deg, rgba(34, 193, 255, 0.2), transparent),
    radial-gradient(circle at center, rgba(255, 255, 255, 0.1), transparent 60%),
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.06) 0, rgba(255, 255, 255, 0.06) 1px, transparent 1px, transparent 7px),
    #050b12;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: rgba(255, 255, 255, 0.8);
}

.video-title {
  font-size: 16px;
  font-weight: 800;
  color: #fff;
}

.video-text {
  margin-top: 8px;
  font-size: 13px;
}

.video-url {
  margin-top: 4px;
  color: rgba(255, 255, 255, 0.45);
  font-size: 12px;
}

.action-group {
  margin-top: 16px;
  display: grid;
  gap: 10px;
}

:deep(.ant-card-head) {
  color: #fff;
  border-bottom-color: rgba(86, 232, 255, 0.18);
}

:deep(.ant-card-body) {
  color: #fff;
}

:deep(.ant-list-item-meta-description) {
  color: rgba(213, 241, 255, 0.58);
}

:deep(.ant-descriptions-item-label) {
  background: rgba(255, 255, 255, 0.04) !important;
  color: rgba(222, 245, 255, 0.72) !important;
}

:deep(.ant-descriptions-item-content) {
  background: rgba(255, 255, 255, 0.02) !important;
  color: #fff !important;
}

:deep(.ant-empty-description) {
  color: rgba(213, 241, 255, 0.58);
}
</style>
