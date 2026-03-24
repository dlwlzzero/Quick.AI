#!/bin/bash
# Build script for CausalLM extension on x86/Linux
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/builddir"
NNTRAINER_ROOT="$PROJECT_ROOT/nntrainer"
NNTRAINER_BUILD="$NNTRAINER_ROOT/builddir_x86"
CAUSALLM_DIR="$NNTRAINER_ROOT/Applications/CausalLM"

echo "=== src - x86 Build ==="
echo "PROJECT_ROOT:   $PROJECT_ROOT"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"

# Check submodule (also verify nested submodules like iniparser are populated)
if [ ! -f "$NNTRAINER_ROOT/meson.build" ] || [ ! -f "$NNTRAINER_ROOT/subprojects/iniparser/src/iniparser.h" ]; then
    echo "Initializing nntrainer submodule (recursive)..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive --depth 1
fi

# ── Step 1: Prepare json.hpp ────────────────────────────────────────────
if [ ! -f "$CAUSALLM_DIR/json.hpp" ]; then
    echo "Preparing json.hpp..."
    pushd "$NNTRAINER_ROOT" > /dev/null
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_BUILD" "0.2" || true
    popd > /dev/null

    # Fallback: manual copy if the script's relative path failed
    if [ ! -f "$CAUSALLM_DIR/json.hpp" ] && [ -f "$NNTRAINER_BUILD/json.hpp" ]; then
        cp "$NNTRAINER_BUILD/json.hpp" "$CAUSALLM_DIR/"
    fi

    if [ ! -f "$CAUSALLM_DIR/json.hpp" ]; then
        echo "Error: Failed to prepare json.hpp"
        exit 1
    fi
fi

# ── Step 2: Build nntrainer core ────────────────────────────────────────
if [ ! -f "$NNTRAINER_BUILD/nntrainer/libnntrainer.so" ]; then
    echo "Building nntrainer..."
    cd "$NNTRAINER_ROOT"

    if [ ! -d "$NNTRAINER_BUILD" ] || [ ! -f "$NNTRAINER_BUILD/build.ninja" ]; then
        rm -rf "$NNTRAINER_BUILD"
        meson setup "$NNTRAINER_BUILD" . \
            --buildtype=release \
            -Denable-app=false \
            -Denable-test=false \
            -Denable-transformer=false \
            -Denable-tflite-backbone=false \
            -Denable-tflite-interpreter=false
    fi

    ninja -C "$NNTRAINER_BUILD" -j $(nproc)
    cd "$SCRIPT_DIR"
else
    echo "nntrainer already built."
fi

# ── Step 3: Build extension ─────────────────────────────────────────────
echo "Building src..."

if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
    rm -rf "$BUILD_DIR"
    meson setup "$BUILD_DIR" "$SCRIPT_DIR" --buildtype=release \
        -Dnntrainer_builddir=builddir_x86
else
    meson setup "$BUILD_DIR" "$SCRIPT_DIR" --reconfigure --buildtype=release \
        -Dnntrainer_builddir=builddir_x86 || true
fi

ninja -C "$BUILD_DIR" -j $(nproc)

echo ""
echo "=== Build completed ==="
echo ""
echo "Run (custom models are built into the executable):"
echo "  LD_LIBRARY_PATH=$NNTRAINER_BUILD/nntrainer:$NNTRAINER_BUILD/api/ccapi:$BUILD_DIR \\"
echo "    $BUILD_DIR/quick_dot_ai <model_path> [input_prompt]"
echo ""
echo "Plugin mode (inject custom models into existing nntr_causallm):"
echo "  LD_PRELOAD=$BUILD_DIR/libquick_dot_ai.so nntr_causallm <model_path>"
