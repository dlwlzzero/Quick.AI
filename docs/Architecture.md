# Quick.AI Native Architecture 🏛️

Quick.AI extends nntrainer's CausalLM application with custom model
implementations, QNN support, XGrammar structured generation, and a deployable
C API.

## 🧱 Native Layers

```text
nntrainer/Applications/CausalLM
  ├── main.cpp, Factory, tokenizer, ChatTemplate
  └── base CausalLM/Transformer implementations

Quick.AI
  ├── src/models/               # self-registering model implementations
  ├── qnn/                      # Android QNN context and SDK wrappers
  ├── src/xgrammar/             # XGrammar manager/wrapper
  └── api/quick_dot_ai_api.*    # handle-based deployment API
```

The native build produces these main artifacts:

| Artifact | Built from | Purpose |
|---|---|---|
| `builddir_*/src/quick_dot_ai` | nntrainer `main.cpp` + Quick.AI static model archive | Standalone runner |
| `builddir_*/src/libquick_dot_ai.so` | Quick.AI model extension objects | `LD_PRELOAD` plugin mode |
| `builddir_*/api/libquick_dot_ai_api.so` | `api/quick_dot_ai_api.cpp` + model deps | Public C API for apps/JNI |
| `builddir_android/qnn/libqnn_context.so` | `qnn/` | QNN context plugin, Android QNN builds |

## 🧩 Self-Registration

Model implementations register themselves with nntrainer's CausalLM factory
before `main()` or API load-time execution:

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

This keeps Quick.AI model additions outside the nntrainer submodule and makes
model implementations independently addable under `src/models/`.

## 🧠 C API Runtime

The public API is `api/quick_dot_ai_api.h`.

- Legacy single-model functions still exist for compatibility.
- New work should use `CausalLmHandle`.
- Each handle owns its model instances, output buffer, metrics, and mutex.
- Different handles can be loaded independently.
- Streaming APIs are synchronous calls that emit token deltas through a callback.

Important entry points:

| API | Purpose |
|---|---|
| `loadModelHandle()` | Load one model handle |
| `runModelHandleStreaming()` | Stream a raw prompt |
| `runModelHandleWithMessagesStreaming()` | Stream OpenAI-style messages |
| `runModelHandleWithJsonStreaming()` | Stream full OpenAI JSON requests |
| `runModelHandleWithTool()` | Run XGrammar-constrained structured generation |
| `runMultimodalHandle*()` | Run image + text paths when supported by the handle |
| `cancelModelHandle()` | Request cooperative cancellation |

## 🧰 Build System

The root `build.sh` prepares nntrainer, tokenizer assets, Android cross files,
and Meson options. `src/` is always built. `api/`, `api-app/`, and `qnn/` are
enabled by build flags:

```bash
./build.sh
./build.sh --platform=android --enable-qnn
./build.sh --target=src,api
```

Meson options live in `meson_options.txt`.

## 📎 Related Docs

- [Main README](../README.md)
- [C API Reference](../api/README.md)
- [Chat Templates](ChatTemplate.md)
- [XGrammar Usage](how-to-use-xgrammar.md)
- [QNN Context Guide](../qnn/README.md)
