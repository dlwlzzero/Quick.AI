# Quick.AI Guides & Examples 🧭

This is the documentation hub for the current Quick.AI repository. Start with
the path that matches what you are building.

## 🚀 Quick Start by User Type

### Android App Developer

Use the `QuickDotAI` AAR directly from an Android app. The current Gradle build
contains `:QuickDotAI` and `:SampleTestAPP`.

| Guide | What you get |
|---|---|
| [QuickDotAI AAR API](../Android/QuickDotAI/README.md) | Kotlin API, model loading, streaming, chat sessions |
| [Android Architecture](../Android/Architecture.md) | Current module layout and planned REST/service layer |
| [Native Streaming](../Android/AsyncAndStreaming.md) | How JNI and the C streaming callback connect |

```kotlin
val engine: QuickDotAI = NativeQuickDotAI(applicationContext)
engine.load(
    LoadModelRequest(
        model = ModelId.GAUSS3_8_QNN,
        backend = BackendType.NPU,
        nativeLibDir = applicationInfo.nativeLibraryDir,
        modelBasePath = "/sdcard/Android/data/com.example.app/files/models"
    )
)

engine.runModelHandleWithMessagesStreaming(
    listOf(
        QuickAiChatMessage(
            role = QuickAiChatRole.USER,
            parts = listOf(PromptPart.Text("Tell me a joke."))
        )
    ),
    sink
)
```

### C/C++ Developer

Use the handle-based C API directly from native applications.

| Guide | What you get |
|---|---|
| [C API Reference](../api/README.md) | Function signatures, enums, error codes, examples |
| [Build Options](../README.md#-building) | Meson flags and Android/x86 build commands |
| [Chat Templates](ChatTemplate.md) | `messages`, `tools`, and `functions` formatting |

```cpp
CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);
runModelHandleStreaming(handle, "Hello!", callback, nullptr);
destroyModelHandle(handle);
```

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
| Structured output and tool calling | [XGrammar Usage](how-to-use-xgrammar.md) |
| OpenAI JSON request streaming | [JSON Streaming API](runWithJsonStreaming_API.md) |
| Chat templates | [Chat Templates](ChatTemplate.md) |
| QNN SDK setup | [QNN Installation](how-to-install-qnn.md) |

## 📋 API References

- [C API](../api/README.md)
- [Android AAR API](../Android/QuickDotAI/README.md)
- [Kotlin DTO source](../Android/QuickDotAI/src/main/java/com/example/quickdotai/Types.kt)

## 🏗️ Architecture & Design

| Document | Topic |
|---|---|
| [Native Architecture](Architecture.md) | Self-registration, model factory, native build outputs |
| [Android Architecture](../Android/Architecture.md) | Current AAR/sample modules and planned REST service |
| [Native Streaming](../Android/AsyncAndStreaming.md) | C callback streaming through JNI |

## 🔗 Quick Links

- [Main README](../README.md)
- [QuickDotAI AAR](../Android/QuickDotAI/README.md)
- [C API Reference](../api/README.md)
- [JSON Streaming API](runWithJsonStreaming_API.md)
