#!/bin/bash
# Build script for CausalLM extension on Android (arm64-v8a)
#
# Usage:
#   ./build_android.sh           # incremental build (reuse existing artifacts)
#   ./build_android.sh --clean   # clean build from scratch
set -e

CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
    esac
done

if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NNTRAINER_ROOT="$PROJECT_ROOT/nntrainer"
CAUSALLM_ROOT="$NNTRAINER_ROOT/Applications/CausalLM"
export NNTRAINER_ROOT

echo "=== src - Android Build ==="
echo "PROJECT_ROOT:   $PROJECT_ROOT"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK:    $ANDROID_NDK"
echo "CLEAN:          $CLEAN"

# Check submodule
if [ ! -f "$NNTRAINER_ROOT/meson.build" ] || [ ! -f "$NNTRAINER_ROOT/subprojects/iniparser/src/iniparser.h" ]; then
    echo "Initializing nntrainer submodule (recursive)..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive --depth 1
fi

# Step 1: nntrainer
if [ "$CLEAN" = true ] || [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    rm -rf builddir
    ./tools/package_android.sh -Dmmap-read=false -Denable-npu=true
else
    echo "nntrainer already built. (use --clean to rebuild)"
fi

if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Error: nntrainer build failed."
    exit 1
fi

# Step 2: Tokenizer
echo "Checking tokenizer library..."
cd "$CAUSALLM_ROOT"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    echo "libtokenizers_android_c.a not found. Building..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        echo "Error: tokenizer library missing and no build script found."
        echo "Place it at: $CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
        exit 1
    fi
fi
echo "Tokenizer library ready."

# Step 3: json.hpp
if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
    echo "json.hpp not found. Downloading..."
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
        echo "Error: Failed to download json.hpp"
        exit 1
    fi
fi

# Step 4: Build src
cd "$SCRIPT_DIR/jni"

if [ "$CLEAN" = true ]; then
    echo "Building src (clean build)..."
    rm -rf libs obj
else
    if [ -f "libs/arm64-v8a/libcausallm.so" ] && [ -f "libs/arm64-v8a/libquick_dot_ai.so" ]; then
        echo "src already built. (use --clean to rebuild)"
        echo "=== Build completed (cached) ==="
        exit 0
    fi
    echo "Building src..."
fi

ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk ENABLE_QNN=1 -j $(nproc)

echo ""
echo "=== Build completed ==="
echo "Output files in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
