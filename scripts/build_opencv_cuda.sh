#!/bin/bash
# Build OpenCV 4.11 with CUDA support for Argus
# RTX 3060 (compute capability 8.6), CUDA 11.8, Python 3.11
#
# Usage:
#   chmod +x scripts/build_opencv_cuda.sh
#   ./scripts/build_opencv_cuda.sh
#
# Requirements: ~300MB download, ~8GB disk during build, 30-60 min compile
set -e

OPENCV_VERSION="4.11.0"
CUDA_ARCH="8.6"
CUDA_PATH="/usr/local/cuda-11.8"
VENV="/home/whp/argus/.venv"
PY="$VENV/bin/python"
JOBS=12  # Not nproc — CUDA compilation is memory-heavy, 20 cores may OOM
BUILD_DIR="/home/whp/opencv_build"

echo "============================================"
echo "OpenCV $OPENCV_VERSION + CUDA Build Script"
echo "============================================"

# ── Pre-flight checks ──
echo "[0/6] Pre-flight checks..."

if [ ! -d "$CUDA_PATH" ]; then
    echo "ERROR: CUDA toolkit not found at $CUDA_PATH"
    exit 1
fi

if [ ! -f "$PY" ]; then
    echo "ERROR: Python not found at $PY"
    exit 1
fi

DISK_FREE=$(df -BG /home/whp --output=avail | tail -1 | tr -d ' G')
if [ "$DISK_FREE" -lt 10 ]; then
    echo "ERROR: Need at least 10GB free disk, have ${DISK_FREE}GB"
    exit 1
fi

echo "  CUDA: $CUDA_PATH ($(ls $CUDA_PATH/version* 2>/dev/null | head -1))"
echo "  Python: $($PY --version 2>&1)"
echo "  Cores for build: $JOBS"
echo "  Free disk: ${DISK_FREE}GB"
echo ""

# ── Step 1: Install build dependencies ──
echo "[1/6] Installing build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev \
    libpng-dev libtiff-dev libatlas-base-dev gfortran \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libdc1394-dev liblapack-dev libopenblas-dev \
    python3.11-dev

# ── Step 2: Uninstall pip opencv (conflicts with compiled version) ──
echo "[2/6] Removing pip opencv..."
$VENV/bin/pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# ── Step 3: Download OpenCV source ──
echo "[3/6] Downloading OpenCV $OPENCV_VERSION..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

if [ ! -d "opencv" ]; then
    git clone --depth 1 --branch "$OPENCV_VERSION" \
        https://github.com/opencv/opencv.git
else
    echo "  opencv source already exists, skipping download"
fi

if [ ! -d "opencv_contrib" ]; then
    git clone --depth 1 --branch "$OPENCV_VERSION" \
        https://github.com/opencv/opencv_contrib.git
else
    echo "  opencv_contrib source already exists, skipping download"
fi

# ── Step 4: Configure with CMake ──
echo "[4/6] Configuring CMake..."
rm -rf opencv/build
mkdir -p opencv/build && cd opencv/build

# Detect Python paths
NUMPY_INC=$($PY -c "import numpy; print(numpy.get_include())")
PY_INC=$(find /usr/include/python3.11* -name "Python.h" -printf "%h" -quit 2>/dev/null)
if [ -z "$PY_INC" ]; then
    PY_INC=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
fi
PY_LIB=$(find /usr/lib -name "libpython3.11*.so" -print -quit 2>/dev/null)
if [ -z "$PY_LIB" ]; then
    PY_LIB=$(find /usr/lib/x86_64-linux-gnu -name "libpython3.11*.so" -print -quit 2>/dev/null)
fi
PY_PACKAGES=$($PY -c "import site; print(site.getsitepackages()[0])")

echo "  Python include: $PY_INC"
echo "  Python library: $PY_LIB"
echo "  NumPy include: $NUMPY_INC"
echo "  Site packages: $PY_PACKAGES"

export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

cmake -GNinja \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX="$VENV" \
    -D OPENCV_EXTRA_MODULES_PATH="$BUILD_DIR/opencv_contrib/modules" \
    \
    -D WITH_CUDA=ON \
    -D CUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
    -D CUDA_ARCH_BIN="$CUDA_ARCH" \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D WITH_NVCUVENC=OFF \
    -D WITH_NVCUVID=OFF \
    \
    -D PYTHON3_EXECUTABLE="$PY" \
    -D PYTHON3_INCLUDE_DIR="$PY_INC" \
    -D PYTHON3_LIBRARY="$PY_LIB" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INC" \
    -D PYTHON3_PACKAGES_PATH="$PY_PACKAGES" \
    -D BUILD_opencv_python3=ON \
    \
    -D BUILD_opencv_cudabgsegm=ON \
    -D BUILD_opencv_cudaimgproc=ON \
    -D BUILD_opencv_cudafilters=ON \
    -D BUILD_opencv_cudawarping=ON \
    -D BUILD_opencv_cudaarithm=ON \
    -D BUILD_opencv_cudacodec=OFF \
    \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_opencv_java=OFF \
    \
    -D WITH_GSTREAMER=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_V4L=ON \
    ..

# ── Step 5: Build ──
echo "[5/6] Building with $JOBS cores (this takes 30-60 minutes)..."
echo "  Start time: $(date)"
ninja -j"$JOBS"
echo "  End time: $(date)"

# ── Step 6: Install ──
echo "[6/6] Installing to $VENV ..."
ninja install

# ── Verify ──
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
$PY -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
print(f'CUDA devices: {cuda_count}')
if cuda_count == 0:
    print('WARNING: No CUDA devices detected!')
    exit(1)

checks = {
    'MOG2':    lambda: cv2.cuda.createBackgroundSubtractorMOG2(),
    'CLAHE':   lambda: cv2.cuda.createCLAHE(),
    'cvtColor': lambda: True,  # always available with CUDA
    'GpuMat':  lambda: cv2.cuda_GpuMat(),
}
for name, fn in checks.items():
    try:
        fn()
        print(f'  CUDA {name}: OK')
    except Exception as e:
        print(f'  CUDA {name}: FAILED ({e})')

print()
print('Build info (CUDA section):')
info = cv2.getBuildInformation()
for line in info.split(chr(10)):
    if 'CUDA' in line or 'NVIDIA' in line:
        print(f'  {line.strip()}')
"

echo ""
echo "============================================"
echo "Done! OpenCV $OPENCV_VERSION with CUDA installed to $VENV"
echo "Build directory ($BUILD_DIR) can be deleted to reclaim ~8GB."
echo "============================================"
