# Quick.AI Guides & Examples 🧭

This is the documentation hub for the current Quick.AI repository. Start with
the path that matches what you are building.

## 🚀 Quick Start by User Type

### Android App Developer

Use the `QuickDotAI` AAR directly from an Android app. The current Gradle build
contains `:QuickDotAI` and `:SampleTestAPP`.

| Guide | What you get |
|---|---|
| [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md) | Chat tab, OpenAI tab, JSON streaming, and XGrammar examples |
| [QuickDotAI AAR API](../Android/QuickDotAI/README.md) | Kotlin API, model loading, streaming, chat sessions |
| [Android Architecture](../Android/Architecture.md) | Current module layout and planned REST/service layer |
| [Android Native Async & Streaming](../Android/AsyncAndStreaming.md) | How JNI and the C streaming callback connect |

For app-level examples, start with the usage guide and use the AAR API
reference for exact type definitions.

### C/C++ Developer

Use the handle-based C API directly from native applications.

| Guide | What you get |
|---|---|
| [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md) | Native messages, JSON streaming, and XGrammar examples |
| [C API Reference](../api/README.md) | Function signatures, enums, and error codes |
| [Build Options](../README.md#-building) | Meson flags and Android/x86 build commands |
| [Chat Templates](ChatTemplate.md) | `messages`, `tools`, and `functions` formatting |

For native examples, start with the usage guide and use the C API reference for
full signatures and error codes.

### Model Developer

Extend Quick.AI with a new CausalLM architecture or QNN model.

| Guide | What you get |
|---|---|
| [Custom Model Guide](../README.md#-how-to-create-a-custom-model) | Model registration and Meson wiring |
| [Native Architecture](Architecture.md) | Plugin system and build artifacts |
| [QNN Context Guide](../qnn/README.md) | QNN backend/context extension details |

## Adding a New Model (T4+)

Starting with T4, adding a model requires only a new translation unit — no
changes to the `ModelType` enum, `loadModelHandle`, or UI code are needed.

1. Create `src/model_descriptors_<name>.cpp` with a `ModelDescriptor` struct
   and an `__attribute__((constructor))` that calls
   `quick_dot_ai::register_model_descriptor(&desc)`. The descriptor fields
   include `id` (string), `family`, `display_name`, `runtime` (0=NATIVE or
   1=LITERT), `backend_mask`, `capabilities`, `config_name`, and
   `arch_string`.
2. Add the new TU to `src/meson.build` so it is linked into
   `libquick_dot_ai_api.so`.
3. The C API catalog (`getModelCatalogJson()`) and the Android `ModelCatalog`
   singleton will automatically reflect the new model after the library is
   rebuilt — no additional registration steps are required.
4. If the model needs a new nntrainer architecture, register it with the
   CausalLM factory as described in the [Custom Model Guide](../README.md#-how-to-create-a-custom-model).

See [Native Architecture](Architecture.md) for the full descriptor struct
layout and the `register_model_descriptor` call convention.

## 🧩 Feature Guides

| Feature | Guide |
|---|---|
| Chat/OpenAI usage examples | [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md) |
| Structured output and tool calling | [XGrammar Reference](XGrammarReference.md) |
| OpenAI JSON request streaming | [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md) |
| Chat templates | [Chat Templates](ChatTemplate.md) |
| QNN SDK setup | [How to Install QNN](HowToInstallQNN.md) |

## 📋 API References

- [C API](../api/README.md)
- [Android AAR API](../Android/QuickDotAI/README.md)
- [Kotlin DTO source](../Android/QuickDotAI/src/main/java/com/example/quickdotai/Types.kt)

## 🏗️ Architecture & Design

| Document | Topic |
|---|---|
| [Native Architecture](Architecture.md) | Self-registration, model factory, native build outputs |
| [Android Architecture](../Android/Architecture.md) | Current AAR/sample modules and planned REST service |
| [Android Native Async & Streaming](../Android/AsyncAndStreaming.md) | C callback streaming through JNI |

## 🔗 Quick Links

- [Main README](../README.md)
- [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md)
- [QuickDotAI AAR](../Android/QuickDotAI/README.md)
- [C API Reference](../api/README.md)
