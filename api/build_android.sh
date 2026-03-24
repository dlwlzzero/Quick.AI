#!/bin/bash
# Build script for src API on Android (arm64-v8a)
#
# Usage:
#   ./build_android.sh           # incremental build (reuse existing artifacts)
#   ./build_android.sh --clean   # clean build from scratch (rebuilds dependencies too)
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

echo "=== src API - Android Build ==="

# Step 1: Build src dependency (pass --clean through)
QUICK_DOT_AI_LIBS="$PROJECT_ROOT/src/jni/libs/arm64-v8a"
if [ "$CLEAN" = true ] || [ ! -f "$QUICK_DOT_AI_LIBS/libcausallm.so" ] || [ ! -f "$QUICK_DOT_AI_LIBS/libquick_dot_ai.so" ]; then
    echo "[1/2] Building src dependency..."
    if [ "$CLEAN" = true ]; then
        "$PROJECT_ROOT/src/build_android.sh" --clean
    else
        "$PROJECT_ROOT/src/build_android.sh"
    fi
else
    echo "[1/2] src already built. (use --clean to rebuild)"
fi

# Step 2: Build api
cd "$SCRIPT_DIR/jni"

if [ "$CLEAN" = true ]; then
    echo "[2/2] Building src API (clean build)..."
    rm -rf libs obj
else
    if [ -f "libs/arm64-v8a/libquick_dot_ai_api.so" ]; then
        echo "[2/2] src API already built. (use --clean to rebuild)"
        echo "=== Build completed (cached) ==="
        exit 0
    fi
    echo "[2/2] Building src API..."
fi

ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)

echo ""
echo "=== Build completed ==="
echo "Output: $SCRIPT_DIR/jni/libs/arm64-v8a/libquick_dot_ai_api.so"
