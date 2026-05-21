#!/bin/bash
echo "=========================================="
echo "  Android Build & Install Script"
echo "=========================================="

# Exit immediately if any command fails
set -e

# ==========================================================
# Configuration
# ==========================================================
APK_APPLICATION="SampleTestApp"

# ==========================================================
# 1. Configure Environment
# ==========================================================
echo "[1/6] Configuring environment variables..."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NDK_ROOT}"
export PATH="${PATH}:${NDK_ROOT}"

if [ -z "$NDK_ROOT" ]; then
    echo "Error: NDK_ROOT environment variable is not set"
    echo "Please set NDK_ROOT to your Android NDK installation path"
    echo "Example: export NDK_ROOT=/path/to/android-ndk"
    exit 1
fi
export ANDROID_NDK="${NDK_ROOT}"
echo "      ANDROID_NDK set to: ${ANDROID_NDK}"

# ==========================================================
# 2. Build NNTrainer for Android with QNN support
# ==========================================================
echo "[2/6] Building project for Android (with QNN, clean build)..."
./build.sh --platform=android --enable-qnn --clean

# ==========================================================
# 3. Install Android Libraries for APK
# ==========================================================
echo "[3/6] Installing Android libraries for APK..."
./apk_install_android.sh

# ==========================================================
# 4. Deploy Prebuilt Libraries
# ==========================================================
echo "[4/6] Copying prebuilt libraries to QuickDotAI project..."
PREBUILT_DIR="./Android/QuickDotAI/prebuilt_libs"

# Ensure destination directory exists
mkdir -p "${PREBUILT_DIR}"

# Copy all shared libraries to the project's prebuilt directory
cp ./install_libs/*.so "${PREBUILT_DIR}/"
echo "      Libraries copied to: ${PREBUILT_DIR}"

# ==========================================================
# 5. Build and Install APK
# ==========================================================
echo "[5/6] Building and installing APK..."
cd ./Android/
./gradlew ":${APK_APPLICATION}:installDebug"

# ==========================================================
# 6. Completion
# ==========================================================
echo "[6/6] Build and installation complete!"
echo "=========================================="
echo "  Success!"
echo "=========================================="
