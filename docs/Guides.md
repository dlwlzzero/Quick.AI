# Quick.AI Guides & Examples

Welcome to the Quick.AI documentation hub. Choose your path below based on your platform and goal.

---

## 🚀 Quick Start by User Type

### Android App Developer
Build Android apps with on-device LLM inference using the QuickDotAI AAR.

| Guide | What you'll learn |
|-------|-----------------|
| [Android AAR Quick Start](../Android/QuickDotAI/README.md) | Gradle setup, Kotlin API, loading models, running inference |
| [Android Architecture](../Android/Architecture.md) | Service architecture, REST API endpoints, component design |
| [Native Streaming](../Android/AsyncAndStreaming.md) | Streaming implementation details, NDJSON, callbacks |

**Quick snippet:**
```kotlin
val engine = NativeQuickDotAI()
engine.load(LoadModelRequest(model = ModelId.GEMMA4, backend = BackendType.GPU))
engine.runStreaming("Tell me a joke.", sink)
engine.close()
```

---

### C/C++ Developer (x86 or Android Native)
Use the C API directly for maximum flexibility and performance.

| Guide | What you'll learn |
|-------|-----------------|
| [C API Reference](../api/README.md) | All functions, enums, structs with examples |
| [README Quick Start](../README.md#c-api-quick-start) | Basic build and run |
| [Build Options](../README.md#build-options) | Meson flags, cross-compilation |

**Quick snippet:**
```cpp
CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);
runModelHandleStreaming(handle, "Hello!", callback, nullptr);
destroyModelHandle(handle);
```

---

### Model Developer (Custom CausalLM)
Extend Quick.AI with your own transformer architecture.

| Guide | What you'll learn |
|-------|-----------------|
| [Custom Model Guide](../README.md#how-to-create-a-custom-model) | Step-by-step plugin authoring |
| [Architecture Overview](Architecture.md) | How the plugin system works |
| [Supported Models](../README.md#supported-models) | Existing models you can reference |

---

## 🔧 Feature Guides

### Structured Generation with XGrammar
Constrain model output to valid JSON schemas for tool calling and structured data extraction.

| Guide | Topic |
|-------|-------|
| [XGrammar Usage](../docs/how-to-use-xgrammar.md) | Integrating XGrammar, schema compilation |
| [JSON Streaming API](../docs/runWithJsonStreaming_API.md) | OpenAI-compatible JSON streaming for tool use |

**See also:** [API function `runModelHandleWithTool()`](../api/README.md#runmodelhandlewithtool)

---

### Chat & Conversations
Use OpenAI-compatible chat templates for multi-turn dialogue.

| Guide | Topic |
|-------|-------|
| [Chat Templates](ChatTemplate.md) | Template discovery, roles (system/user/assistant), tool prompts |

**See also:** [API function `runModelHandleWithMessages()`](../api/README.md#runmodelhandlewithmessages)

---

### QNN / NPU Acceleration
Run models on Qualcomm NPU (HTP) for maximum efficiency on Android devices.

| Guide | Topic |
|-------|-------|
| [QNN Installation](how-to-install-qnn.md) | QNN SDK and Hexagon SDK setup |
| [QNN Context Development](../qnn/README.md) | Custom QNN backend contexts |
| [Build with QNN](../README.md#android-arm64-v8a) | `--enable-qnn` build instructions |

---

## 📋 API References

### C API
The complete C API specification with every function, enum, struct, and error code.

- [Full API Reference](../api/README.md)
- [Error Codes](../api/README.md#error-code-reference)
- [Model Types](../api/README.md#11-supported-models)

### Android AAR (Kotlin)
Public API surface of the `QuickDotAI` module.

- [AAR API](../Android/QuickDotAI/README.md)
- [Types & DTOs](../Android/QuickDotAI/src/main/java/com/example/quickdotai/Types.kt) (source)

---

## 🏗️ Architecture & Design

Deep dives into internal design decisions.

| Document | Topic |
|----------|-------|
| [Plugin Architecture](Architecture.md) | Self-registration mechanism, Factory singleton, build artifacts |
| [Android Service Architecture](../Android/Architecture.md) | QuickAIService, ModelRegistry, ModelWorker, REST endpoints |
| [Streaming Design](../Android/AsyncAndStreaming.md) | NDJSON streaming, ChunkedStreamSink, LiteRT-LM integration |

---

## 🔗 Quick Links from README

- [Main README](../README.md) — Project overview, features, directory structure
- [Supported Models](../README.md#supported-models) — Gauss-2.5, Gauss-3, QNN models
- [Build System](../README.md#building) — `build.sh`, `install_android.sh`, `apk-build-install.sh`
