#!/bin/bash
# Build script for src API test app on Android (arm64-v8a)
#
# Usage:
#   ./build_android.sh           # incremental build (reuse existing artifacts)
#   ./build_android.sh --clean   # clean build from scratch (rebuilds all dependencies)
set -e

CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
    esac
done

if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NNTRAINER_ROOT="$PROJECT_ROOT/nntrainer"
export NNTRAINER_ROOT

INCLUDE_DIR="$SCRIPT_DIR/include"

echo "=== src API Test App - Android Build ==="

# Step 1: Build dependency chain (src → api)
API_LIB="$PROJECT_ROOT/api/jni/libs/arm64-v8a/libquick_dot_ai_api.so"
if [ "$CLEAN" = true ] || [ ! -f "$API_LIB" ]; then
    echo "[1/2] Building api dependency chain..."
    if [ "$CLEAN" = true ]; then
        "$PROJECT_ROOT/api/build_android.sh" --clean
    else
        "$PROJECT_ROOT/api/build_android.sh"
    fi
else
    echo "[1/2] api already built. (use --clean to rebuild)"
fi

# Step 2: Copy API header and build test app
mkdir -p "$INCLUDE_DIR"
cp "$PROJECT_ROOT/api/quick_dot_ai_api.h" "$INCLUDE_DIR/"

cd "$SCRIPT_DIR/jni"

if [ "$CLEAN" = true ]; then
    echo "[2/2] Building test application (clean build)..."
    rm -rf libs obj
else
    echo "[2/2] Building test application..."
fi

ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)

echo ""
echo "=== Build completed ==="
echo "Output: $SCRIPT_DIR/jni/libs/arm64-v8a/quick_dot_ai_test"
echo ""
echo "Install to device:"
echo "  ./install_android.sh"