#!/bin/bash
# Installation script for src API Test App on Android device
set -e

INSTALL_DIR="/data/local/tmp/Quick.AI"
MODEL_DIR="$INSTALL_DIR/models"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected. Please connect a device and try again."
    exit 1
fi

NNTRAINER_ROOT="$PROJECT_ROOT/nntrainer"

# Check if test app was built
if [ ! -f "$SCRIPT_DIR/jni/libs/arm64-v8a/quick_dot_ai_test" ]; then
    echo "Error: quick_dot_ai_test not found. Please run build_android.sh first."
    exit 1
fi

echo "Installing src API Test App to Android device..."

# Create directories on device
adb shell "mkdir -p $INSTALL_DIR"
adb shell "mkdir -p $MODEL_DIR"

# Push base shared libraries (nntrainer, causallm, etc.)
echo "Pushing base shared libraries..."
adb push "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" $INSTALL_DIR/
adb push "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so" $INSTALL_DIR/
adb push "$PROJECT_ROOT/src/jni/libs/arm64-v8a/libc++_shared.so" $INSTALL_DIR/
adb push "$PROJECT_ROOT/src/jni/libs/arm64-v8a/libcausallm.so" $INSTALL_DIR/

echo "Pushing libquick_dot_ai.so..."
adb push "$PROJECT_ROOT/src/jni/libs/arm64-v8a/libquick_dot_ai.so" $INSTALL_DIR/

# Push API library
echo "Pushing libquick_dot_ai_api.so..."
adb push "$PROJECT_ROOT/api/jni/libs/arm64-v8a/libquick_dot_ai_api.so" $INSTALL_DIR/

# Push test executable
echo "Pushing test executable..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/quick_dot_ai_test" $INSTALL_DIR/
adb shell "chmod 755 $INSTALL_DIR/quick_dot_ai_test"

# Create run script on device
adb shell "cat > $INSTALL_DIR/run_test.sh << 'RUNEOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/Quick.AI:\$LD_LIBRARY_PATH
cd /data/local/tmp/Quick.AI
./quick_dot_ai_test \$@
RUNEOF"

adb shell "chmod 755 $INSTALL_DIR/run_test.sh"

echo ""
echo "=== Installation completed ==="
echo ""
echo "Usage:"
echo "  adb shell $INSTALL_DIR/run_test.sh <model_name> [prompt] [chat_template] [quant] [verbose]"
echo ""
echo "Arguments:"
echo "  model_name      Model name: qwen3-0.6b, gauss2.5-1b"
echo "  prompt          Input prompt (default: \"Hello, how are you?\")"
echo "  chat_template   true/false (default: true)"
echo "  quant           W4A32/W16A16/W8A16/W32A32 (default: W4A32)"
echo "  verbose         true/false (default: true)"
echo ""
echo "Examples:"
echo "  adb shell $INSTALL_DIR/run_test.sh qwen3-0.6b"
echo "  adb shell $INSTALL_DIR/run_test.sh gauss2.5-1b \"What is AI?\" true W4A32"
echo ""
echo "Note: Push model files before running:"
echo "  adb push <model_dir>/ $MODEL_DIR/"
