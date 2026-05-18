# Quick.AI ⚡

Quick.AI is an on-device LLM stack built around nntrainer CausalLM extensions.
It provides self-registering C++ model plugins, a handle-based C API, Qualcomm
QNN integration, and an Android AAR (`QuickDotAI`) with native and LiteRT-LM
backends.

## 📚 Table of Contents

- [Features](#-features)
- [Supported Models](#-supported-models)
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Building](#-building)
- [How to Create a Custom Model](#-how-to-create-a-custom-model)
- [Architecture](#-architecture)
- [Directory Structure](#-directory-structure)
- [Documentation](#-documentation)

## ✨ Features

- **Self-registering model plugins**: add custom `CausalLM` models without
  changing nntrainer source files.
- **Handle-based C API**: load independent model handles, stream tokens, cancel
  in-flight runs, collect metrics, and use OpenAI-style messages.
- **Android AAR**: `QuickDotAI` exposes `NativeQuickDotAI` for nntrainer/QNN
  models and `LiteRTLm` for Gemma-family `.litertlm` models.
- **Structured generation**: XGrammar-backed tool/schema constrained output via
  `runModelHandleWithTool()`.
- **Chat templates**: OpenAI-compatible `messages`, `tools`, and `functions`
  formatting through model-local `chat_template.jinja` or
  `tokenizer_config.json`.
- **Multimodal paths**: LiteRT-LM image input for Gemma-family models and native
  QNN vision paths where the loaded model supplies vision + LLM sub-models.

## 🤖 Supported Models

The C API model enum is defined in [`api/quick_dot_ai_api.h`](api/quick_dot_ai_api.h).
Android `ModelId` values are defined in
[`Android/QuickDotAI/src/main/java/com/example/quickdotai/Types.kt`](Android/QuickDotAI/src/main/java/com/example/quickdotai/Types.kt).

| C enum | Android `ModelId` | Notes |
|---|---|---|
| `CAUSAL_LM_MODEL_QWEN3_0_6B` | `QWEN3_0_6B` | Native nntrainer model |
| `CAUSAL_LM_MODEL_GAUSS2_5` | currently native-only | Built-in C API config |
| `CAUSAL_LM_MODEL_GAUSS3_6_QNN` | `GAUSS3_6_QNN` | Android QNN |
| `CAUSAL_LM_MODEL_GAUSS3_8_QNN` | `GAUSS3_8_QNN` | Android QNN |
| `CAUSAL_LM_MODEL_QWEN3_1_7B_Q40` | `QWEN3_1_7B_Q40` | Native nntrainer model |
| `CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN` | `GAUSS3_8_VISION_QNN` | Native QNN vision model |
| `CAUSAL_LM_MODEL_GAUSS3_6` | `GAUSS3_6` | Native nntrainer model |
| `CAUSAL_LM_MODEL_TINY_BERT` | `TINY_BERT` | Native model |
| `CAUSAL_LM_MODEL_FUNCTION_GEMMA` | `FUNCTION_GEMMA` | Tool-calling oriented model |
| `CAUSAL_LM_MODEL_GAUSS3_8` | `GAUSS3_8` | Native nntrainer model |
| `CAUSAL_LM_MODEL_GEMMA4_CPU` | `GEMMA4_CPU` | Native CPU Gemma path |
| `CAUSAL_LM_MODEL_GEMMA4_E2B_QNN` | `GEMMA4_E2B_QNN` | Android QNN |
| Kotlin-only | `GEMMA4` | Routed to `LiteRTLm`; requires a `.litertlm` path |

Model configuration files are placed under `src/res/` and model
implementations live under `src/models/`.

## 🚀 Quick Start

### Android AAR

Quick.AI currently ships the `QuickDotAI` AAR module and the direct
`SampleTestAPP` sample. The REST/foreground-service layer described in
older plans is not part of the current Gradle build.

```kotlin
dependencies {
    implementation(project(":QuickDotAI"))
}
```

See [`docs/ChatAndOpenAIUsage.md`](docs/ChatAndOpenAIUsage.md) for Chat tab,
OpenAI tab, JSON streaming, and XGrammar examples. See
[`Android/QuickDotAI/README.md`](Android/QuickDotAI/README.md) for the full AAR
API.

### C API

```cpp
#include "quick_dot_ai_api.h"

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

runModelHandleStreaming(handle, "Hello!", [](const char *delta, void *) {
  std::cout << delta << std::flush;
  return 0;
}, nullptr);

destroyModelHandle(handle);
```

See [`api/README.md`](api/README.md) for the complete C API reference.

## 🧰 Prerequisites

- C++17 compiler
- [Meson](https://mesonbuild.com/) >= 0.55.0
- [Ninja](https://ninja-build.org/)
- Android NDK for Android builds
- OpenBLAS for x86 builds (`apt install libopenblas-dev`)
- nntrainer submodule dependencies
- Qualcomm QNN and Hexagon SDK for `--enable-qnn` Android builds

## 🏗️ Building

All native builds go through the root `build.sh`.

### x86 / Linux

```bash
./build.sh
./build.sh --target=src
./build.sh --clean
```

Run the standalone executable:

```bash
LD_LIBRARY_PATH=nntrainer/builddir_x86/nntrainer:nntrainer/builddir_x86/api/ccapi:builddir_x86/src:builddir_x86/api \
  builddir_x86/src/quick_dot_ai /path/to/model "Your prompt"
```

Plugin mode is still available for the original `nntr_causallm` executable:

```bash
LD_PRELOAD=$(pwd)/builddir_x86/src/libquick_dot_ai.so nntr_causallm /path/to/model
```

### Android Arm64 Build

```bash
export ANDROID_NDK=/path/to/android-ndk

./build.sh --platform=android
./build.sh --platform=android --enable-qnn
./install_android.sh
```

To build native libraries, copy them into `Android/QuickDotAI/prebuilt_libs/`,
and install `SampleTestAPP`:

```bash
./apk-build-install.sh
```

Before running `apk-build-install.sh`, set `NDK_ROOT` inside the script to your
local Android NDK path.

### Build Options

| Option | Default | Description |
|---|---|---|
| `--platform=x86|android` | `x86` | Target platform |
| `--target=src,api,api-test,qnn` | `all` | Comma-separated target set |
| `--enable-qnn` | off | Enable QNN integration (Android only) |
| `--clean` | off | Clean rebuild |

Meson options are declared in [`meson_options.txt`](meson_options.txt).

## 🧩 How to Create a Custom Model

1. Add your implementation under `src/models/<model_name>/`.
2. Inherit from `causallm::CausalLM` or the appropriate Quick.AI model base.
3. Register the architecture in your `.cpp` file:

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

4. Add model config files under `src/res/<model_name>/`.
5. Add `src/models/<model_name>/meson.build` and include it from
   `src/models/meson.build`.

## 🏛️ Architecture

Quick.AI uses nntrainer's CausalLM application and `Factory` registration, while
Quick.AI-specific models are compiled into `quick_dot_ai` and
`libquick_dot_ai.so`. The deployable C API is `libquick_dot_ai_api.so`.

See [`docs/Architecture.md`](docs/Architecture.md) for native architecture and
[`Android/Architecture.md`](Android/Architecture.md) for Android module status.

## 🗂️ Directory Structure

```text
project-root/
├── nntrainer/              # nntrainer submodule
├── xgrammar/               # XGrammar submodule
├── src/                    # Native CausalLM extensions and model configs
├── api/                    # libquick_dot_ai_api.so public C API
├── qnn/                    # Android QNN context library
├── Android/
│   ├── QuickDotAI/         # Android AAR
│   └── SampleTestAPP/      # Direct sample app
├── docs/                   # Canonical project documentation
├── gemma_python/           # Gemma4 quantization-oriented Python package
├── build.sh
├── install_android.sh
└── apk-build-install.sh
```

## 📖 Documentation

| Document | Audience | Content |
|---|---|---|
| [`docs/Guides.md`](docs/Guides.md) | All users | Entry points by platform and goal |
| [`docs/ChatAndOpenAIUsage.md`](docs/ChatAndOpenAIUsage.md) | App/API users | Chat tab, OpenAI tab, JSON streaming, and XGrammar examples |
| [`docs/Architecture.md`](docs/Architecture.md) | Native contributors | Plugin, build, and C API architecture |
| [`api/README.md`](api/README.md) | C/C++ users | C API reference |
| [`Android/QuickDotAI/README.md`](Android/QuickDotAI/README.md) | Android users | AAR API surface and types |
| [`Android/Architecture.md`](Android/Architecture.md) | Android contributors | Current modules and planned service layer |
| [`docs/ChatTemplate.md`](docs/ChatTemplate.md) | Model/API users | Chat template discovery and JSON request handling |
| [`docs/XGrammarReference.md`](docs/XGrammarReference.md) | Tool-calling users | XGrammar internals, toolsets, cache behavior, and native API notes |
| [`qnn/README.md`](qnn/README.md) | QNN developers | QNN context development guide |
