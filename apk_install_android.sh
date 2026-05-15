#!/bin/bash
# APK 빌드용 라이브러리 설치 스크립트
# install_libs/ 디렉토리에 라이브러리만 복사 (adb push 없음)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/builddir_android"
NNTRAINER_ROOT="$SCRIPT_DIR/nntrainer"
NNTRAINER_ANDROID="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a"
INSTALL_LIBS_DIR="$SCRIPT_DIR/install_libs"

# ── Validate ────────────────────────────────────────────────────────────
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Run './build.sh --platform=android' first."
    exit 1
fi

echo "=== Installing libraries for APK build ==="
echo "Install dir: $INSTALL_LIBS_DIR"
echo ""

mkdir -p "$INSTALL_LIBS_DIR"

# ── Copy nntrainer runtime libraries ────────────────────────────────────
echo "Copying nntrainer libraries..."
[ -f "$NNTRAINER_ANDROID/libnntrainer.so" ] && cp "$NNTRAINER_ANDROID/libnntrainer.so" "$INSTALL_LIBS_DIR/"
[ -f "$NNTRAINER_ANDROID/libccapi-nntrainer.so" ] && cp "$NNTRAINER_ANDROID/libccapi-nntrainer.so" "$INSTALL_LIBS_DIR/"

# ── Copy built artifacts ────────────────────────────────────────────────
echo "Copying built artifacts..."

# src targets
for f in libcausallm.so libquick_dot_ai.so; do
    [ -f "$BUILD_DIR/src/$f" ] && cp "$BUILD_DIR/src/$f" "$INSTALL_LIBS_DIR/"
done

[ -f "$BUILD_DIR/src/quick_dot_ai" ] && cp "$BUILD_DIR/src/quick_dot_ai" "$INSTALL_LIBS_DIR/"

# api target
[ -f "$BUILD_DIR/api/libquick_dot_ai_api.so" ] && cp "$BUILD_DIR/api/libquick_dot_ai_api.so" "$INSTALL_LIBS_DIR/"

# api-test target
[ -f "$BUILD_DIR/api-app/quick_dot_ai_test" ] && cp "$BUILD_DIR/api-app/quick_dot_ai_test" "$INSTALL_LIBS_DIR/"

# qnn target
[ -f "$BUILD_DIR/qnn/libqnn_context.so" ] && cp "$BUILD_DIR/qnn/libqnn_context.so" "$INSTALL_LIBS_DIR/"

# ── Copy libc++_shared.so from NDK ──────────────────────────────────────
if [ -n "$ANDROID_NDK" ]; then
    LIBCXX=$(find "$ANDROID_NDK" -name "libc++_shared.so" -path "*/aarch64*" 2>/dev/null | head -1)
    if [ -n "$LIBCXX" ]; then
        echo "Copying libc++_shared.so..."
        cp "$LIBCXX" "$INSTALL_LIBS_DIR/"
    fi
fi

echo ""
echo "=== Installation completed ==="
echo "Libraries copied to: $INSTALL_LIBS_DIR"
echo ""
echo "Copied files:"
ls -la "$INSTALL_LIBS_DIR/"