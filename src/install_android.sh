#!/bin/bash
# Installation script for src on Android device
set -e

INSTALL_DIR="/data/local/tmp/Quick.AI"
MODEL_DIR="$INSTALL_DIR/models"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NNTRAINER_ROOT="$PROJECT_ROOT/nntrainer"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected. Please connect a device and try again."
    exit 1
fi

# Check if build was successful
if [ ! -f "$SCRIPT_DIR/jni/libs/arm64-v8a/quick_dot_ai_exe" ]; then
    echo "Error: quick_dot_ai_exe not found. Please run build_android.sh first."
    exit 1
fi

echo "Installing src to Android device..."

# Create directories on device
adb shell "mkdir -p $INSTALL_DIR"
adb shell "mkdir -p $MODEL_DIR"

# Push executable
echo "Pushing executable..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/quick_dot_ai_exe" $INSTALL_DIR/quick_dot_ai
adb shell "chmod 755 $INSTALL_DIR/quick_dot_ai"

# Push shared libraries
echo "Pushing shared libraries..."
adb push "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" $INSTALL_DIR/
adb push "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so" $INSTALL_DIR/
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so" $INSTALL_DIR/
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so" $INSTALL_DIR/

# Push custom plugin (optional, for LD_PRELOAD with original nntr_causallm)
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libquick_dot_ai.so" ]; then
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libquick_dot_ai.so" $INSTALL_DIR/
fi

# Create run script on device
adb shell "cat > $INSTALL_DIR/run.sh << 'RUNEOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/Quick.AI:\$LD_LIBRARY_PATH
cd /data/local/tmp/Quick.AI
./quick_dot_ai \$@
RUNEOF"

adb shell "chmod 755 $INSTALL_DIR/run.sh"

echo ""
echo "=== Installation completed ==="
echo ""
echo "To run on device:"
echo "  1. Push model files:"
echo "     adb push res/gauss_2_5/ $MODEL_DIR/gauss_2_5/"
echo ""
echo "  2. Run:"
echo "     adb shell $INSTALL_DIR/run.sh $MODEL_DIR/gauss_2_5"
echo ""
echo "  3. Interactive shell:"
echo "     adb shell"
echo "     cd $INSTALL_DIR"
echo "     ./run.sh $MODEL_DIR/gauss_2_5 \"Your prompt here\""
