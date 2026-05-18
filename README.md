# Quick.AI⚡

Custom model extensions for [nntrainer](https://github.com/nntrainer/nntrainer) CausalLM application.

Build your own CausalLM models as **self-registering plugins** — no modification to nntrainer's source code required.

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Quick Start](#quick-start)
  - [Android Quick Start](#android-quick-start)
  - [C API Quick Start](#c-api-quick-start)
- [Prerequisites](#prerequisites)
- [Building](#building)
  - [x86 / Linux](#x86--linux)
  - [Android (arm64-v8a)](#android-arm64-v8a)
  - [Build Options](#build-options)
- [How to Create a Custom Model](#how-to-create-a-custom-model)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Documentation](#documentation)

## Features

Quick.AI provides a complete stack for on-device LLM inference, from low-level C++ plugins to high-level Android APIs.

- **Self-Registration Plugin System**: Add custom `CausalLM` models without modifying nntrainer
- **C API**: Production-ready handle-based API with streaming, multimodal (vision), and XGrammar structured generation
- **Android AAR**: `QuickDotAI` library with dual backends:
  - `NativeQuickDotAI` — NNTrainer-based backend (Qwen3, Gauss, etc.)
  - `LiteRTLm` — Google LiteRT-LM backend (Gemma family, multimodal)
- **XGrammar Integration**: JSON-schema constrained output for 100% structurally correct generation
- **Chat Templates**: OpenAI-compatible message formatting (`system`/`user`/`assistant`) with tool/function calling support
- **Multimodal**: Image + text input (vision encoder + LLM) via QNN on Android

## Supported Models

| Model | Directory | Architecture Key | Platform | Notes |
|---|---|---|---|---|
| **Gauss-2.5** | `src/models/gauss-2.5/` | `Gauss2_5ForCausalLM` | x86, Android | Base model with standard attention/MLP |
| **Gauss-3** | `src/models/gauss-3/` | `Gauss3ForCausalLM` | x86, Android | Sliding window attention, dynamic theta |
| **QNN Models** | `src/models/qnn/` | `Gauss3_6ForCausalLM` / `Gauss3_8ForCausalLM` | Android (QNN) | NPU-accelerated via Qualcomm QNN SDK |

Model configs are placed in `src/res/<model_name>/`:
- **config.json**: Model architecture and weights mapping
- **generation_config.json**: Token IDs, sampling parameters
- **nntr_config.json**: NNTrainer runtime settings (tensor types, sequence lengths)

## Quick Start

### Android Quick Start

Quick.AI distributes a foreground service (`QuickAIService`) and an AAR (`QuickDotAI`) for Android apps.

```kotlin
// Gradle dependency
implementation(project(":QuickDotAI"))
```

```kotlin
val engine: QuickDotAI = when (req.model) {
    ModelId.GEMMA4 -> LiteRTLm(applicationContext)
    else           -> NativeQuickDotAI()
}

engine.load(LoadModelRequest(model = ModelId.GEMMA4, backend = BackendType.GPU))
engine.runStreaming("Tell me a joke.", sink)
engine.close()
```

- See [Android/QuickDotAI/README.md](Android/QuickDotAI/README.md) for the full AAR API
- See [Android/Architecture.md](Android/Architecture.md) for service architecture and REST endpoints
- See [Android/AsyncAndStreaming.md](Android/AsyncAndStreaming.md) for native streaming design

### C API Quick Start

The C API (`api/quick_dot_ai_api.h`) supports handle-based multi-model inference.

```cpp
#include "quick_dot_ai_api.h"

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

// Streaming inference
runModelHandleStreaming(handle, "Hello!", [](const char *delta, void *) {
    std::cout << delta << std::flush;
    return 0; // 0 = continue, non-zero = cancel
}, nullptr);

destroyModelHandle(handle);
```

- See [api/README.md](api/README.md) for the complete API reference

## Prerequisites

- C++17 compiler
- [Meson](https://mesonbuild.com/) >= 0.55.0
- [Ninja](https://ninja-build.org/)
- [Android NDK](https://developer.android.com/ndk) (for android builds)
- OpenBLAS (for x86 builds: `apt install libopenblas-dev`)
- nntrainer dependencies (see nntrainer documentation)

## Building

All builds are driven by the unified `build.sh` script at the project root.

### x86 / Linux

```bash
# Build all targets (src + api + api-test)
./build.sh

# Build only src (model library + executable)
./build.sh --target=src

# Clean rebuild
./build.sh --clean
```

Two ways to run:

```bash
# Standalone executable (recommended — custom models built in)
LD_LIBRARY_PATH=nntrainer/builddir_x86/nntrainer:nntrainer/builddir_x86/api/ccapi:builddir_x86/src:builddir_x86/api \
  builddir_x86/src/quick_dot_ai /path/to/model "Your prompt"

# Plugin mode (inject into existing nntr_causallm via LD_PRELOAD)
LD_PRELOAD=$(pwd)/builddir_x86/src/libquick_dot_ai.so nntr_causallm /path/to/model
```

### Android (arm64-v8a)

**Option A: Native library only (for C++ development / testing)**

```bash
export ANDROID_NDK=/path/to/android-ndk

# Build all targets
./build.sh --platform=android

# Build with QNN support (android only)
./build.sh --platform=android --enable-qnn

# Install native libraries to device
./install_android.sh

# Run standalone executable directly on device
adb shell /data/local/tmp/Quick.AI/run.sh /path/to/model
```

**Option B: Full APK build (for Android app development)**

```bash
# Build native libs, copy to AAR, build & install APK
./apk-build-install.sh
```

> **Note**: `./apk-build-install.sh` performs the full Android APK workflow:
> 1. Builds the project with QNN support (`./build.sh --platform=android --enable-qnn --clean`)
> 2. Installs native libraries (`./apk_install_android.sh`)
> 3. Copies `.so` files to `Android/QuickDotAI/prebuilt_libs/`
> 4. Builds and installs the debug APK via Gradle
>
> **Before running**, you must edit `apk-build-install.sh` to match your environment:
> - Set `NDK_ROOT` to your Android NDK path (default is hardcoded)
> - Adjust build flags (`--enable-qnn`, `--clean`, etc.) as needed
>
> **Difference from Option A**: Option B builds the full Android application (LauncherApp/SampleTestAPP) with Gradle and installs the APK. Option A only builds and installs native libraries for direct command-line execution via `adb shell`.

### Build Options

| Option | Default | Description |
|---|---|---|
| `--platform=x86\|android` | `x86` | Target platform |
| `--target=src,api,api-test,qnn` | `all` | Comma-separated list of targets |
| `--enable-qnn` | off | Enable QNN integration (android only) |
| `--clean` | off | Clean rebuild from scratch |

Meson options (set via `-D` or in `meson_options.txt`):

| Option | Default | Description |
|---|---|---|
| `platform` | `auto` | Target platform (`auto`, `x86`, `android`) |
| `enable-qnn` | `false` | Build QNN context lib + qnn-transformer model (android only) |
| `enable-fp16` | `true` | Enable FP16 support (effective on android/ARM only) |
| `enable-api` | `false` | Build `libquick_dot_ai_api.so` |
| `enable-api-test` | `false` | Build `quick_dot_ai_test` executable |

> **Note**: When using `./build.sh` without `--target`, both `enable-api` and `enable-api-test` are automatically enabled. Use `--target=src` to disable them.

## How to Create a Custom Model

### 1. Define Your Model Class

Inherit from `causallm::CausalLM` (see `models/gauss-2.5/gauss2_5_causallm.h`):

```
Transformer          (base: embedding + decoder blocks + norm)
    ├── CausalLM     (adds LM head + generation logic)
    └── Gauss2_5Transformer  (customize attention/MLP)
         └── Gauss2_5CausalLM  (combines both)
```

Key virtual methods to override:
- `createAttention()` — Q/K/V projections, MHA configuration
- `createMlp()` — Feed-forward network
- `createTransformerDecoderBlock()` — Full decoder block
- `registerCustomLayers()` — Register custom nntrainer layers

### 2. Self-Register in the `.cpp` File

At the bottom of your `.cpp` file, add:

```cpp
__attribute__((constructor)) static void register_my_models() {
  causallm::Factory::Instance().registerModel(
    "MyModelForCausalLM",
    [](causallm::json cfg, causallm::json generation_cfg,
       causallm::json nntr_cfg) {
      return std::make_unique<causallm::MyModel>(
        cfg, generation_cfg, nntr_cfg);
    });
}
```

### 3. Configure Your Model

Create config files in `res/your_model/`:
- **config.json**: Set `"architectures": ["MyModelForCausalLM"]` (must match registered key)
- **generation_config.json**: Token IDs, sampling parameters
- **nntr_config.json**: NNTrainer settings (tensor types, sequence lengths, etc.)

### 4. Add to Build System

For a new model `models/my_model/`:

1. Create `models/my_model/meson.build`:
```meson
my_model_src = [meson.current_source_dir() / 'my_model.cpp']
my_model_inc = include_directories('.')
quick_dot_ai_src += my_model_src
quick_dot_ai_inc += my_model_inc
```

2. Add `subdir('my_model')` to `models/meson.build`

## Architecture

Quick.AI uses a self-registering plugin system built on nntrainer.
See [docs/Architecture.md](docs/Architecture.md) for the full architecture 
diagram and technical details.

## Directory Structure

```
project-root/
├── nntrainer/                          # Shared nntrainer submodule (untouched)
├── meson_options.txt                    # Build options (platform, enable-qnn, etc.)
├── build.sh                            # Unified build script (x86 + android)
├── install_android.sh                  # Unified android device installation
├── cross/
│   └── android-aarch64.cross.in        # NDK cross-compilation template
├── xgrammar/                           # XGrammar submodule (structured generation)
├── Android/                            # Android application and AAR
│   ├── QuickDotAI/                     # QuickDotAI AAR module
│   ├── SampleTestAPP/                  # Sample Android test application
│   └── Architecture.md                 # Android architecture documentation
├── src/                                # CausalLM custom model extension
│   ├── models/                         # Model implementations (see Supported Models below)
│   ├── res/                            # Model configuration files (JSON configs)
│   └── meson.build                     # src subdir build
├── qnn/                                # QNN context library (android only)
│   ├── qnn_context.cpp
│   ├── jni/                            # QNN SDK wrappers + RPC manager
│   └── meson.build
├── api/                                # C API for deploying models
│   ├── quick_dot_ai_api.h
│   ├── quick_dot_ai_api.cpp
│   ├── model_config.cpp
│   └── meson.build
├── api-app/                            # API test application
│   ├── test_api.cpp
│   └── meson.build
└── install_libs/                       # Pre-built/shared libraries output
```

## Documentation

| Document | Target Audience | Content |
|---|---|---|
| [docs/Guides.md](docs/Guides.md) | All users | Platform-specific quick starts, feature guides, API references |
| [docs/Architecture.md](docs/Architecture.md) | Contributors/Developers | Plugin system architecture and design |
| [api/README.md](api/README.md) | C/C++ developers | Full C API specification with examples |
| [Android/Architecture.md](Android/Architecture.md) | Android developers | Service architecture and REST endpoints |
| [qnn/README.md](qnn/README.md) | QNN developers | QNN context development guide |

**Quick links by topic:**
- [Structured generation (XGrammar)](docs/how-to-use-xgrammar.md)
- [Chat templates](docs/ChatTemplate.md)
- [JSON streaming API](docs/runWithJsonStreaming_API.md)
- [QNN installation](docs/how-to-install-qnn.md)
- [Android streaming design](Android/AsyncAndStreaming.md)
