#!/bin/bash
# Installation script for src API on Android device
set -e

INSTALL_DIR="/data/local/tmp/Quick.AI"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected. Please connect a device and try again."
    exit 1
fi

# Check if build was successful
if [ ! -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libquick_dot_ai_api.so" ]; then
    echo "Error: libquick_dot_ai_api.so not found. Please run build_android.sh first."
    exit 1
fi

echo "Installing src API to Android device..."

# Create directory on device
adb shell "mkdir -p $INSTALL_DIR"

# Push API library
echo "Pushing libquick_dot_ai_api.so..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libquick_dot_ai_api.so" $INSTALL_DIR/

echo ""
echo "=== Installation completed ==="
echo "Installed: libquick_dot_ai_api.so -> $INSTALL_DIR/"
echo ""
echo "NOTE: Make sure src/install_android.sh was run first"
echo "      (installs nntrainer, causallm, and other dependencies)."
