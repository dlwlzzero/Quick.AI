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
