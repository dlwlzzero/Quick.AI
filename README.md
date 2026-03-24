# Quick.AI⚡

You can run your model with Quick.AI everywhere.
Quick.AI is custom model extensions for [nntrainer](https://github.com/nntrainer/nntrainer) CausalLM application.
Build your own CausalLM models as **self-registering plugins** — no modification to nntrainer's source code required.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  nntrainer/Applications/CausalLM (submodule)                 │
│  ├── main.cpp + Factory singleton                            │
│  │     ├── Qwen3ForCausalLM      (built-in)                  │
│  │     ├── GptOssForCausalLM     (built-in)                  │
│  │     └── ...                                               │
│  │                                                           │
│  src                                                         |
│  ├── models            (standalone executable)               │
│  │     └── Gauss2_5ForCausalLM ◄── statically linked (always │
│  │                                 available, no LD_PRELOAD) │
│  └── libquick_dot_ai.so          (plugin for LD_PRELOAD)     │
│        __attribute__((constructor)) runs before main()       │
│        → Factory::Instance().registerModel(...)              │
└──────────────────────────────────────────────────────────────┘
```

Custom models use `__attribute__((constructor))` to register with the `Factory` singleton **before `main()` starts**. The standalone executable (`quick_dot_ai`) statically links the custom models via `link_whole`, so they are always available without `LD_PRELOAD`. A shared library is also built for plugin mode with the original `nntr_causallm`.

This means:

- nntrainer's `main.cpp` is used as-is — never copied or modified
- When nntrainer updates, nothing in this repo breaks
- Multiple custom models can be added independently

## Directory Structure

```
Quick.AI/
├── nntrainer/                          # Shared nntrainer submodule (untouched)
├── src/                       # CausalLM custom model extension
│   ├── models/
│   │   ├── meson.build                 # Lists model subdirectories
│   │   ├── gauss-2.5/                  # Gauss-2.5 model implementation
│   │   │   ├── gauss2_5_causallm.h
│   │   │   ├── gauss2_5_causallm.cpp  # Includes __attribute__((constructor))
│   │   │   └── meson.build
│   │   └── gauss-3.6/                  # Gauss-3.6 model implementation
│   │       ├── gauss3_6_causallm.h
│   │       ├── gauss3_6_causallm.cpp
│   │       └── meson.build
│   ├── res/gauss_2_5/                  # Gauss-2.5 model configuration
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   └── nntr_config.json
│   ├── jni/                            # Android NDK build files
│   │   ├── Android.mk
│   │   └── Application.mk
│   ├── meson.build                     # Meson build (x86/Linux)
│   ├── build_x86.sh
│   ├── build_android.sh
│   └── install_android.sh
├── api/                   # C API for deploying models
│   ├── quick_dot_ai_api.h
│   ├── quick_dot_ai_api.cpp
│   └── model_config.cpp
├── api-app/          # API test application
│   └── test_api.cpp
└── ...
```

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
3. Add source and include paths to `jni/Android.mk`

## Building

### x86 / Linux

```bash
cd src
./build_x86.sh
```

Two ways to run:

```bash
# Standalone executable (recommended — custom models built in)
LD_LIBRARY_PATH=../nntrainer/builddir_x86/nntrainer:../nntrainer/builddir_x86/api/ccapi:./builddir \
  ./builddir/quick_dot_ai /path/to/model "Your prompt"

# Plugin mode (inject into existing nntr_causallm via LD_PRELOAD)
LD_PRELOAD=$(pwd)/builddir/libquick_dot_ai.so nntr_causallm /path/to/model
```

### Android (arm64-v8a)

```bash
cd src
export ANDROID_NDK=/path/to/android-ndk
./build_android.sh
./install_android.sh

# Run on device
adb shell /data/local/tmp/nntrainer/quick_dot_ai/run.sh /path/to/model
```

## Prerequisites

- C++17 compiler
- [Meson](https://mesonbuild.com/) >= 0.55.0 (for x86 build)
- [Android NDK](https://developer.android.com/ndk) (for Android build)
- nntrainer dependencies (see nntrainer documentation)



