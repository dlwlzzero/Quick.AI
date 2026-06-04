# Quick.AI C API Reference 🧠

The Quick.AI C API is declared in `quick_dot_ai_api.h` and implemented by
`libquick_dot_ai_api.so`. It exposes nntrainer-backed model loading,
handle-based inference, streaming callbacks, multimodal paths, QNN KV-cache
helpers, chat templates, and XGrammar structured generation.

## 📚 Contents

- [Model Catalog (T4)](#-model-catalog-t4)
- [Model Enums](#-model-enums)
- [Core Types](#-core-types)
- [Global Options](#-global-options)
- [Legacy Single Model API](#-legacy-single-model-api)
- [Handle Based API](#-handle-based-api)
- [Streaming](#-streaming)
- [Multimodal](#-multimodal)
- [XGrammar](#-xgrammar)
- [OpenAI JSON Streaming](#-openai-json-streaming)
- [Error Codes](#-error-codes)

## Model Catalog (T4)

As of T4, the preferred model identification mechanism is a **string model id**
routed through a self-registering descriptor catalog.

### loadModelHandleByName

```c
ErrorCode loadModelHandleByName(BackendType compute,
                                const char *model_id,
                                ModelQuantizationType quant_type,
                                const char *native_lib_dir,
                                const char *model_base_path,
                                CausalLmHandle *out_handle);
```

`model_id` is a string such as `"qwen3-0.6b"` or `"gemma4-cpu"`. The function
looks up the descriptor in the process-global registry, resolves the config,
and loads the model. This is the **preferred T4 load path** for all new code.

### getModelCatalogJson

```c
const char *getModelCatalogJson(void);
```

Returns a JSON array of all model descriptors registered in the current
process. The returned pointer is valid for the lifetime of the process. Example
output:

```json
[
  {
    "id": "qwen3-0.6b",
    "family": "qwen3-0.6b",
    "display_name": "Qwen3 0.6B",
    "runtime": 0,
    "backend_mask": 3,
    "capabilities": 5,
    "config_name": "qwen3_0_6b",
    "arch_string": "Qwen3ForCausalLM"
  },
  {
    "id": "gemma4-cpu",
    "family": "gemma4-cpu",
    "display_name": "Gemma4 CPU",
    "runtime": 0,
    "backend_mask": 1,
    "capabilities": 1,
    "config_name": "gemma4_cpu",
    "arch_string": "Gemma3ForCausalLM"
  }
]
```

`runtime` is `0` for `NATIVE` and `1` for `LITERT`. `backend_mask` is a
bitmask where bit 0 = CPU, bit 1 = GPU, bit 2 = NPU/QNN. `capabilities` is a
bitmask using the `Capability` enum flags (bit 0 = STREAMING, bit 1 =
MESSAGES_API, bit 2 = MULTIMODAL, bit 3 = TOOL_USE, bit 4 = EMBEDDING).

### ModelDescriptor struct

```c
typedef struct {
  const char *id;
  const char *family;
  const char *display_name;
  int         runtime;       // 0 = NATIVE, 1 = LITERT
  uint32_t    backend_mask;
  uint32_t    capabilities;
  const char *config_name;
  const char *arch_string;
} ModelDescriptor;
```

## 🤖 Model Enums

> **Deprecated.** The `ModelType` enum is a compatibility shim maintained for
> ABI stability. Gauss entries have been removed. All new code should use
> string model ids with `loadModelHandleByName()` instead.

| Enum | Value | Notes |
|---|---:|---|
| `CAUSAL_LM_MODEL_QWEN3_0_6B` | 0 | Qwen3 0.6B |
| `CAUSAL_LM_MODEL_GAUSS2_5` | 1 | Gauss 2.5 |
| `CAUSAL_LM_MODEL_GAUSS3_6_QNN` | 2 | Gauss 3.6 QNN |
| `CAUSAL_LM_MODEL_GAUSS3_8_QNN` | 3 | Gauss 3.8 QNN |
| `CAUSAL_LM_MODEL_QWEN3_1_7B_Q40` | 4 | Qwen3 1.7B Q40 |
| `CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN` | 5 | Gauss 3.8 vision encoder QNN |
| `CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN` | 6 | Gauss 3.8 vision/ViT QNN |
| `CAUSAL_LM_MODEL_GAUSS3_6` | 7 | Gauss 3.6 |
| `CAUSAL_LM_MODEL_TINY_BERT` | 8 | TinyBERT |
| `CAUSAL_LM_MODEL_FUNCTION_GEMMA` | 9 | Function-calling Gemma |
| `CAUSAL_LM_MODEL_GAUSS3_8` | 10 | Gauss 3.8 |
| `CAUSAL_LM_MODEL_GEMMA4_CPU` | 11 | Gemma4 CPU |
| `CAUSAL_LM_MODEL_GEMMA4_E2B_QNN` | 12 | Gemma4 E2B QNN |

QNN models require Android builds with `--enable-qnn`.

## 🧱 Core Types

```c
typedef struct CausalLmModel *CausalLmHandle;

typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1,
  CAUSAL_LM_BACKEND_NPU = 2,
} BackendType;

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

typedef struct {
  const char *role;
  const char *content;
} CausalLMChatMessage;

typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
```

`delta` passed to `CausalLmTokenCallback` is valid only during the callback.
Copy it if you need to keep it.

## ⚙️ Global Options

```c
typedef struct {
  bool use_chat_template;
  bool debug_mode;
  bool verbose;
  const char *chat_template_name;
} Config;

ErrorCode setOptions(Config config);
```

`setOptions()` affects global chat-template/debug behavior for subsequent API
calls in the current process.

## 🕰️ Legacy Single Model API

These functions operate on one process-wide default handle. Prefer the
handle-based API for new code.

```c
ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type,
                    const char *model_base_path);

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);

ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                            size_t num_messages,
                            bool add_generation_prompt,
                            const char **formattedText);

ErrorCode saveQnnKvCache(const char *cache_path);
ErrorCode loadQnnKvCache(const char *cache_path);
ErrorCode resetQnnKvCache(void);
```

## 🧩 Handle Based API

```c
ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          const char *native_lib_dir,
                          const char *model_base_path,
                          CausalLmHandle *out_handle);

ErrorCode destroyModelHandle(CausalLmHandle handle);
ErrorCode unloadModelHandle(CausalLmHandle handle);
ErrorCode cancelModelHandle(CausalLmHandle handle);

ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics);

ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path);
ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path);
ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle);
```

`native_lib_dir` is mainly used by Android/QNN flows to locate shared
libraries. `model_base_path` is the base directory for model files.

For native QNN backend extensions, set
`QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH` before loading the model to override
the default `htp_backend_ext_config.json` location. Absolute values are used
as-is. Relative values are resolved by `QNNContext` from
`QUICK_DOT_AI_BASE_DIR` when set, otherwise from the process current working
directory. If no override is set, the C API uses
`<model_base_path without trailing /models>/htp_backend_ext_config.json`.

## 🔄 Streaming

```c
ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                  const char *inputTextPrompt,
                                  CausalLmTokenCallback callback,
                                  void *user_data);

ErrorCode runModelHandleWithMessages(
  CausalLmHandle handle,
  const CausalLMChatMessage *messages,
  size_t num_messages,
  bool add_generation_prompt,
  const char **outputText);

ErrorCode runModelHandleWithMessagesStreaming(
  CausalLmHandle handle,
  const CausalLMChatMessage *messages,
  size_t num_messages,
  bool add_generation_prompt,
  CausalLmTokenCallback callback,
  void *user_data);
```

Streaming calls are synchronous. They block the calling thread until generation
finishes, fails, or is cancelled, while progressively invoking `callback`.

### Minimal Streaming Example

```cpp
#include "quick_dot_ai_api.h"
#include <iostream>

int on_token(const char *delta, void *) {
  std::cout << delta << std::flush;
  return 0;
}

int main() {
  CausalLmHandle handle = nullptr;
  ErrorCode err = loadModelHandle(CAUSAL_LM_BACKEND_NPU,
                                  CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                                  CAUSAL_LM_QUANTIZATION_W4A32,
                                  nullptr,
                                  "/models",
                                  &handle);
  if (err != CAUSAL_LM_ERROR_NONE) return err;

  err = runModelHandleStreaming(handle, "Hello!", on_token, nullptr);
  destroyModelHandle(handle);
  return err;
}
```

## 🖼️ Multimodal

```c
ErrorCode runMultimodalHandleStreaming(
  CausalLmHandle handle,
  const char *prompt,
  const float *pixelValues,
  int numPatches,
  int originalHeight,
  int originalWidth,
  CausalLmTokenCallback callback,
  void *user_data);

ErrorCode runMultimodalHandleWithMessages(
  CausalLmHandle handle,
  const CausalLMChatMessage *messages,
  size_t num_messages,
  bool add_generation_prompt,
  const float *pixelValues,
  int numPatches,
  int originalHeight,
  int originalWidth,
  const char **outputText);

ErrorCode runMultimodalHandleWithMessagesStreaming(
  CausalLmHandle handle,
  const CausalLMChatMessage *messages,
  size_t num_messages,
  bool add_generation_prompt,
  const float *pixelValues,
  int numPatches,
  int originalHeight,
  int originalWidth,
  CausalLmTokenCallback callback,
  void *user_data);
```

The handle must be loaded from a model configuration that supplies the expected
vision encoder + LLM sub-models. Unsupported handles return
`CAUSAL_LM_ERROR_UNSUPPORTED`.

## 🧰 XGrammar

```c
ErrorCode runModelHandleWithTool(CausalLmHandle handle,
                                 const char *inputTextPrompt,
                                 const char **outputText,
                                 const char *tool_name,
                                 const char *tool_schema);
```

If a model directory contains `Toolset.json`, tools are precompiled at model
load. For dynamic schemas, pass `tool_schema` on first use.

See [`../docs/ChatAndOpenAIUsage.md`](../docs/ChatAndOpenAIUsage.md) for usage
examples and [`../docs/XGrammarReference.md`](../docs/XGrammarReference.md)
for XGrammar internals.

## 📡 OpenAI JSON Streaming

```c
ErrorCode runModelHandleWithJsonStreaming(CausalLmHandle handle,
                                          const char *jsonRequest,
                                          CausalLmTokenCallback callback,
                                          void *user_data);
```

`jsonRequest` accepts OpenAI-style request JSON, including `messages`, `tools`,
and legacy `functions`. A chat template must be available from the loaded model
directory or the call returns `CAUSAL_LM_ERROR_UNSUPPORTED`.

See [`../docs/ChatAndOpenAIUsage.md`](../docs/ChatAndOpenAIUsage.md) for usage
examples and request routing details.

## ❌ Error Codes

| Code | Constant | Meaning |
|---:|---|---|
| 0 | `CAUSAL_LM_ERROR_NONE` | Success |
| 1 | `CAUSAL_LM_ERROR_INVALID_PARAMETER` | Null pointer, invalid request, bad JSON, or invalid argument |
| 2 | `CAUSAL_LM_ERROR_MODEL_LOAD_FAILED` | Model/config/weight load failed |
| 3 | `CAUSAL_LM_ERROR_INFERENCE_FAILED` | Runtime inference failure |
| 4 | `CAUSAL_LM_ERROR_NOT_INITIALIZED` | Handle/model is not loaded |
| 5 | `CAUSAL_LM_ERROR_INFERENCE_NOT_RUN` | Metrics requested before inference |
| 6 | `CAUSAL_LM_ERROR_UNSUPPORTED` | Feature unsupported by this build/model/handle |
| 99 | `CAUSAL_LM_ERROR_UNKNOWN` | Unknown internal failure |

## 📎 Related Docs

- [Main README](../README.md)
- [Android AAR API](../Android/QuickDotAI/README.md)
- [Chat and OpenAI Usage Examples](../docs/ChatAndOpenAIUsage.md)
- [Chat Templates](../docs/ChatTemplate.md)
- [XGrammar Reference](../docs/XGrammarReference.md)
