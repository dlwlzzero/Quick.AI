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

## 🧩 Self Registration

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
| `loadModelHandleByName()` | Preferred load path — load one handle by string model id |
| `loadModelHandle()` | Legacy load path using the deprecated `ModelType` enum |
| `loadMultimodalHandleByName()` | Pair a vision/embedding model with an LLM into one handle |
| `runModelHandleStreaming()` | Stream a raw prompt |
| `runModelHandleWithMessagesStreaming()` | Stream OpenAI-style messages |
| `runModelHandleWithJsonStreaming()` | Stream full OpenAI JSON requests |
| `runModelHandleWithTool()` | Run XGrammar-constrained structured generation |
| `runMultimodalHandle*()` | Run single-image + text paths when supported by the handle |
| `runMultimodalMultiImageHandle*()` | Run multi-image paths (e.g. `vjepa-qnn`) |
| `cancelModelHandle()` | Request cooperative cancellation |
| `destroyModelHandle()` | Release handle resources |
| `unloadModelHandle()` | Unload model (handle remains valid) |
| `getPerformanceMetricsHandle()` | Get per-handle performance metrics |
| `saveQnnKvCacheHandle()` | Save QNN KV cache |
| `loadQnnKvCacheHandle()` | Load QNN KV cache |
| `resetQnnKvCacheHandle()` | Reset QNN KV cache |

## Model Registry

The API layer maintains a **string-keyed self-registering model descriptor
catalog** separate from the nntrainer CausalLM factory.

### Self-registration

A descriptor is a `ModelDescriptor` struct (declared in
`api/model_descriptor.h`) registered into a process-global registry at load
time via `quick_dot_ai::register_model_descriptor(&desc)`:

```cpp
static const ModelDescriptor desc = {
  "qwen3-0.6b",        // id
  "qwen3-0.6b",        // family
  "Qwen3 0.6B",        // display_name
  QDA_RUNTIME_NATIVE,  // runtime (QDA_RUNTIME_NATIVE = 0, QDA_RUNTIME_LITERT = 1)
  B(0) | B(1),         // backend_mask (CPU | GPU)
  QDA_CAP_STREAMING | QDA_CAP_TOOL_USE, // capabilities bitmask
  "QWEN3-0.6B",        // config_name (key into the internal config registry)
  "Qwen3ForCausalLM",  // arch_string (nntrainer Factory key)
};

__attribute__((constructor)) static void register_descriptors() {
  quick_dot_ai::register_model_descriptor(&desc);
}
```

This runs before `main()` / API first-call, adding the descriptor to the
registry. No central switch statement or header change is needed.

Two registration sites exist in the tree:

- **Built-in catalog** — descriptors for the public models (Qwen3, TinyBERT,
  Function-Gemma, Gemma4, LFM2-VL, and the QNN models) live together in
  `api/model_descriptors_public.cpp`, registered by a single constructor. QNN
  descriptors there are guarded by `ENABLE_QNN`.
- **Plugin models** — a model implementation under `src/models/<name>/` may
  register its own descriptor inline in the same `__attribute__((constructor))`
  that registers it with the nntrainer `Factory` (see the QNN models under
  `src/models/qnn/`).

`src/model_descriptor_stub.cpp` provides a **weak no-op**
`register_model_descriptor()` so the standalone `quick_dot_ai` executable links
even when the API library (which carries the strong definition) is not present.

> Note: there are no `src/model_descriptors_<name>.cpp` files — descriptors are
> consolidated in `api/model_descriptors_public.cpp` or inlined in each model's
> own translation unit.

### Catalog API

| Function | Purpose |
|---|---|
| `loadModelHandleByName(backend, model_id, quant, lib_dir, base_path, out)` | Preferred load path — routes through the descriptor registry |
| `getModelCatalogJson()` | Returns a JSON array of all registered descriptors |

`getModelCatalogJson()` returns a JSON array in this shape:

```json
[
  {
    "id": "qwen3-0.6b",
    "family": "qwen3-0.6b",
    "display_name": "Qwen3 0.6B",
    "runtime": 0,
    "backend_mask": 3,
    "capabilities": 9
  }
]
```

Only these six fields are serialized. `config_name` and `arch_string` are
internal descriptor fields used to resolve the config and the Factory
architecture; they are **not** exposed in the catalog JSON. `backend_mask` is a
bitmask (bit 0 = CPU, bit 1 = GPU, bit 2 = NPU/QNN); `capabilities` is a
`CapabilityFlag` bitmask (bit 0 = STREAMING, 1 = MESSAGES_API, 2 = MULTIMODAL,
3 = TOOL_USE, 4 = EMBEDDING, 5 = MULTI_IMAGE, 6 = VISION_ENCODER).

### ModelType enum status

The `CAUSAL_LM_MODEL_*` C enum is a **deprecated compatibility shim**.
Ordinals are preserved for ABI compatibility, so the values are not
contiguous. All new code should use string model ids and
`loadModelHandleByName()`.

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
- [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md)
- [Chat Templates](ChatTemplate.md)
- [XGrammar Reference](XGrammarReference.md)
- [QNN Context Guide](../qnn/README.md)
