# Quick.AI API Specification

Quick.AI is a C API library for on-device AI model execution. It supports LLM inference with text generation and multimodal (image+text) processing capabilities.

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Supported Models](#11-supported-models)
   - [1.2 Backend Types](#12-backend-types)
   - [1.3 Quantization Types](#13-quantization-types)
2. [API Reference](#2-api-reference)
   - [2.1 Configuration and Initialization](#21-configuration-and-initialization)
   - [2.2 Legacy Single Model API](#22-legacy-single-model-api)
   - [2.3 Handle-based API - Model Management](#23-handle-based-api---model-management)
   - [2.4 Handle-based API - Inference Execution](#24-handle-based-api---inference-execution)
   - [2.5 Multimodal API](#25-multimodal-api)
   - [2.6 XGrammar API - Structured Generation](#26-xgrammar-api---structured-generation)
3. [Usage Guide](#3-usage-guide)
   - [3.1 C/C++ Examples](#31-cc-examples)
   - [3.2 Kotlin/Java (JNI) Usage Guide](#32-kotlinjava-jni-usage-guide)
4. [Error Code Reference](#4-error-code-reference)

---

## 1. Overview

### 1.1 Supported Models

| Enum Value | Integer | Model Name | Description |
|------------|---------|------------|-------------|
| `CAUSAL_LM_MODEL_QWEN3_0_6B` | 0 | Qwen3 0.6B | Qwen3 0.6B model |
| `CAUSAL_LM_MODEL_GAUSS2_5` | 1 | Gauss 2.5 | Gauss 2.5 1B model |
| `CAUSAL_LM_MODEL_GAUSS3_6_QNN` | 2 | Gauss 3.6 QNN | Gauss 3.6 QNN accelerated model (QNN required) |
| `CAUSAL_LM_MODEL_GAUSS3_8_QNN` | 3 | Gauss 3.8 QNN | Gauss 3.8 QNN accelerated model (QNN required) |
| `CAUSAL_LM_MODEL_QWEN3_1_7B_Q40` | 4 | Qwen3 1.7B Q40 | Qwen3 1.7B Q40 quantized model |
| `CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN` | 5 | Gauss 3.8 VE QNN | Gauss 3.8 Vision Encoder (QNN required) |
| `CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN` | 6 | Gauss 3.8 ViT QNN | Gauss 3.8 ViT model (QNN required) |
| `CAUSAL_LM_MODEL_GAUSS3_6` | 7 | Gauss 3.6 | Gauss 3.6 model (non-QNN) |
| `CAUSAL_LM_MODEL_TINY_BERT` | 8 | TinyBERT | Multilingual TinyBERT model |
| `CAUSAL_LM_MODEL_FUNCTION_GEMMA` | 9 | Function Gemma | Function calling Gemma model |
| `CAUSAL_LM_MODEL_GAUSS3_8` | 10 | Gauss 3.8 | Gauss 3.8 model (non-QNN) |
| `CAUSAL_LM_MODEL_GEMMA4_CPU` | 11 | Gemma4 CPU | Gemma 4 CPU model |

> **Note**: Models with QNN suffix are only available when `ENABLE_QNN` build option is enabled.

### 1.2 Backend Types

| Type | Description |
|------|-------------|
| `CAUSAL_LM_BACKEND_CPU` | CPU backend |
| `CAUSAL_LM_BACKEND_GPU` | GPU backend (coming soon) |
| `CAUSAL_LM_BACKEND_NPU` | NPU backend (QNN required) |

### 1.3 Quantization Types

| Type | Description |
|------|-------------|
| `CAUSAL_LM_QUANTIZATION_UNKNOWN` | Default |
| `CAUSAL_LM_QUANTIZATION_W4A32` | 4-bit Weight, 32-bit Activation |
| `CAUSAL_LM_QUANTIZATION_W16A16` | 16-bit Weight, 16-bit Activation |
| `CAUSAL_LM_QUANTIZATION_W8A16` | 8-bit Weight, 16-bit Activation |
| `CAUSAL_LM_QUANTIZATION_W32A32` | 32-bit Weight, 32-bit Activation |

---

## 2. API Reference

### 2.1 Configuration and Initialization

| API | Description |
|---|---|
| [setOptions()](#setoptions) | Sets global configuration options (chat template, debug, verbose) |

#### setOptions()

```c
ErrorCode setOptions(Config config);
```

Sets global options.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Config` | Configuration struct |

**Config Struct Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `use_chat_template` | `bool` | Whether to apply chat template |
| `debug_mode` | `bool` | Debug mode (validates model files) |
| `verbose` | `bool` | Verbose output |
| `chat_template_name` | `const char *` | Template name ("default", "tool_use", etc.) |

**Returns**: `ErrorCode`

---

### 2.2 Legacy Single Model API

> **Note**: These APIs use a global single model instance. Use handle-based APIs to run multiple models simultaneously.

| API | Description |
|---|---|
| [loadModel()](#loadmodel) | Loads a model into the global single instance |
| [getPerformanceMetrics()](#getperformancemetrics) | Retrieves performance metrics from the last inference |
| [applyChatTemplate()](#applychattemplate) | Applies chat template to chat messages without running inference |
| [saveQnnKvCache()](#saveqnnkvcache) | Saves QNN KV cache to file |
| [loadQnnKvCache()](#loadqnnkvcache) | Loads QNN KV cache from file |
| [resetQnnKvCache()](#resetqnnkvcache) | Resets QNN KV cache |

#### loadModel()

```c
ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type,
                    const char *model_base_path);
```

Loads a model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `compute` | `BackendType` | Backend type |
| `modeltype` | `ModelType` | Model type |
| `quant_type` | `ModelQuantizationType` | Quantization type |
| `model_base_path` | `const char *` | Model base path (NULL for default path) |

**Returns**: `ErrorCode`

---

#### getPerformanceMetrics()

```c
ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);
```

Retrieves performance metrics from the last inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `PerformanceMetrics *` | Pointer to receive metrics struct |

**Returns**: `ErrorCode`

---

#### applyChatTemplate()

```c
ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                            size_t num_messages,
                            bool add_generation_prompt,
                            const char **formattedText);
```

Applies chat template to chat messages.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `const CausalLMChatMessage *` | Message array |
| `num_messages` | `size_t` | Number of messages |
| `add_generation_prompt` | `bool` | Whether to add generation prompt |
| `formattedText` | `const char **` | Pointer to receive formatted text |

**CausalLMChatMessage Struct Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `role` | `const char *` | Message role ("system", "user", "assistant") |
| `content` | `const char *` | Message content |

**Returns**: `ErrorCode`

> **Note**: `formattedText` points to an internal library buffer and is valid until the next call.

---

#### saveQnnKvCache()

```c
ErrorCode saveQnnKvCache(const char *cache_path);
```

Saves QNN KV cache.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_path` | `const char *` | Cache file path |

**Returns**: `ErrorCode`

> **Note**: Only supported in QNN builds.

---

#### loadQnnKvCache()

```c
ErrorCode loadQnnKvCache(const char *cache_path);
```

Loads QNN KV cache.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_path` | `const char *` | Cache file path |

**Returns**: `ErrorCode`

> **Note**: Only supported in QNN builds.

---

#### resetQnnKvCache()

```c
ErrorCode resetQnnKvCache(void);
```

Resets QNN KV cache.

**Returns**: `ErrorCode`

> **Note**: Only supported in QNN builds.

---

### 2.3 Handle-based API - Model Management

> **Tip**: Handle-based APIs allow loading multiple models simultaneously and running them in parallel. Each handle has independent state.

| API | Description |
|---|---|
| [loadModelHandle()](#loadmodelhandle) | Loads a model and returns a handle |
| [destroyModelHandle()](#destroymodelhandle) | Releases the handle and associated resources |
| [unloadModelHandle()](#unloadmodelhandle) | Unloads the model from a handle (keeps handle struct) |
| [getPerformanceMetricsHandle()](#getperformancemetricshandle) | Retrieves per-handle performance metrics |
| [saveQnnKvCacheHandle()](#saveqnnkvcachehandle) | Saves QNN KV cache for a handle |
| [loadQnnKvCacheHandle()](#loadqnnkvcachehandle) | Loads QNN KV cache for a handle |
| [resetQnnKvCacheHandle()](#resetqnnkvcachehandle) | Resets QNN KV cache for a handle |

#### loadModelHandle()

```c
ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          const char *native_lib_dir,
                          const char *model_base_path,
                          CausalLmHandle *out_handle);
```

Loads a model and returns a handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `compute` | `BackendType` | Backend type |
| `modeltype` | `ModelType` | Model type |
| `quant_type` | `ModelQuantizationType` | Quantization type |
| `native_lib_dir` | `const char *` | Native library path (for Android, can be NULL) |
| `model_base_path` | `const char *` | Model base path |
| `out_handle` | `CausalLmHandle *` | Pointer to receive the created handle |

**Returns**: `ErrorCode`

---

#### destroyModelHandle()

```c
ErrorCode destroyModelHandle(CausalLmHandle handle);
```

Releases the handle and associated resources.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Handle to release (can be NULL) |

**Returns**: `ErrorCode`

> **Note**: Passing NULL handle returns success without any operation.

---

#### unloadModelHandle()

```c
ErrorCode unloadModelHandle(CausalLmHandle handle);
```

Unloads the model from a handle (keeps handle structure).

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |

**Returns**: `ErrorCode`

---

#### getPerformanceMetricsHandle()

```c
ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics);
```

Retrieves per-handle performance metrics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `metrics` | `PerformanceMetrics *` | Pointer to receive metrics struct |

**Returns**: `ErrorCode`

---

#### saveQnnKvCacheHandle()

```c
ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path);
```

Saves QNN KV cache for a handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `cache_path` | `const char *` | Cache file path |

**Returns**: `ErrorCode`

---

#### loadQnnKvCacheHandle()

```c
ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path);
```

Loads QNN KV cache for a handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `cache_path` | `const char *` | Cache file path |

**Returns**: `ErrorCode`

---

#### resetQnnKvCacheHandle()

```c
ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle);
```

Resets QNN KV cache for a handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |

**Returns**: `ErrorCode`

---

### 2.4 Handle-based API - Inference Execution

> **Tip**: Inference functions are categorized into blocking and streaming modes.

| API | Description |
|---|---|
| [runModelHandleWithMessages()](#runmodelhandlewithmessages) | Blocking inference with OpenAI message format |
| [runModelHandleStreaming()](#runmodelhandlestreaming) | Streaming inference with raw text prompt |
| [cancelModelHandle()](#cancelmodelhandle) | Cancels ongoing inference on a handle |
| [runModelHandleWithMessagesStreaming()](#runmodelhandlewithmessagesstreaming) | Streaming inference with OpenAI message format |
| [runModelHandleWithJsonStreaming()](#runmodelhandlewithjsonstreaming) | Streaming inference with OpenAI JSON format |

#### Streaming API Details

Streaming APIs invoke a callback function for each generated token, delivering results in real-time.

**Callback Function Signature**:
```c
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
```

| Parameter | Description |
|-----------|-------------|
| `delta` | UTF-8 text generated at current token. Valid only during callback invocation; copy if needed |
| `user_data` | User data pointer passed to streaming functions |

**Return Values**:
- `0`: Continue generation
- `Non-zero`: Request cancellation (stops at next token boundary)

**Streaming vs Blocking Comparison**:

| Feature | Blocking API | Streaming API |
|---------|-------------|---------------|
| Result delivery | Once after completion | Real-time per token |
| Memory | Requires full result buffer | Token-by-token processing |
| Cancellation | Requires `cancelModelHandle()` call | Immediate via callback return value |
| UX | Long wait time | Real-time response feel |
| Use case | Batch processing, API servers | Chat UI, real-time generation |

**Streaming Best Practices**:
1. The `delta` pointer is valid only during callback invocation. Copy data if you need to retain it.
2. Heavy operations in callbacks may slow down generation speed.
3. Do not throw exceptions from callbacks (C API compatibility).
4. Concurrent calls on the same handle are internally serialized.

#### runModelHandleWithMessages()

```c
ErrorCode runModelHandleWithMessages(CausalLmHandle handle,
                                     const CausalLMChatMessage *messages,
                                     size_t num_messages,
                                     bool add_generation_prompt,
                                     const char **outputText);
```

Runs inference with OpenAI message format (blocking).

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `messages` | `const CausalLMChatMessage *` | Message array |
| `num_messages` | `size_t` | Number of messages |
| `add_generation_prompt` | `bool` | Whether to add generation prompt |
| `outputText` | `const char **` | Pointer to receive output text |

**Returns**: `ErrorCode`

> **Note**: `outputText` is owned by the handle and valid until the next inference or handle destruction.

**Example (C++)**:
```cpp
CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

CausalLMChatMessage messages[] = {
    {.role = "user", .content = "Hello!"}
};
const char *output = nullptr;
ErrorCode err = runModelHandleWithMessages(handle, messages, 1, true, &output);
if (err == CAUSAL_LM_ERROR_NONE) {
    printf("Output: %s\n", output);
}
destroyModelHandle(handle);
```

---

#### runModelHandleStreaming()

```c
ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                  const char *inputTextPrompt,
                                  CausalLmTokenCallback callback,
                                  void *user_data);
```

Runs inference in streaming mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `inputTextPrompt` | `const char *` | Input prompt (UTF-8) |
| `callback` | `CausalLmTokenCallback` | Token callback function |
| `user_data` | `void *` | User data to pass to callback |

**Returns**: `ErrorCode`

**Example (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0; // 0: continue, non-zero: cancel
}

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

runModelHandleStreaming(handle, "Tell me a story.", token_callback, nullptr);
destroyModelHandle(handle);
```

---

#### cancelModelHandle()

```c
ErrorCode cancelModelHandle(CausalLmHandle handle);
```

Cancels ongoing inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |

**Returns**: `ErrorCode`

> **Note**: Thread-safe, can be called from any thread.

---

#### runModelHandleWithMessagesStreaming()

```c
ErrorCode runModelHandleWithMessagesStreaming(CausalLmHandle handle,
                                              const CausalLMChatMessage *messages,
                                              size_t num_messages,
                                              bool add_generation_prompt,
                                              CausalLmTokenCallback callback,
                                              void *user_data);
```

Runs streaming inference with OpenAI message format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `messages` | `const CausalLMChatMessage *` | Message array |
| `num_messages` | `size_t` | Number of messages |
| `add_generation_prompt` | `bool` | Whether to add generation prompt |
| `callback` | `CausalLmTokenCallback` | Token callback function |
| `user_data` | `void *` | User data to pass to callback |

**Returns**: `ErrorCode`

**Example (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0; // 0: continue
}

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

CausalLMChatMessage messages[] = {
    {.role = "user", .content = "What is AI?"}
};
runModelHandleWithMessagesStreaming(handle, messages, 1, true, token_callback, nullptr);
destroyModelHandle(handle);
```

---

#### runModelHandleWithJsonStreaming()

```c
ErrorCode runModelHandleWithJsonStreaming(CausalLmHandle handle,
                                          const char *jsonRequest,
                                          CausalLmTokenCallback callback,
                                          void *user_data);
```

Runs streaming inference with OpenAI JSON format. Parses the JSON request (including messages, tools, and functions) and applies the chat template, then drives generation token-by-token.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `jsonRequest` | `const char *` | OpenAI format JSON string (UTF-8, NUL-terminated) |
| `callback` | `CausalLmTokenCallback` | Token callback function |
| `user_data` | `void *` | User data to pass to callback |

**Returns**: `ErrorCode`

**Example JSON Input**:
```json
{
  "messages": [
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "call", "description": "..."}}
  ]
}
```

---

### 2.5 Multimodal API

> **Prerequisite**: Multimodal APIs are only supported in QNN builds. The handle must be loaded with Vision Encoder and LLM sub-models.

| API | Description |
|---|---|
| [runMultimodalHandleStreaming()](#runmultimodalhandlestreaming) | Streaming multimodal (image+text) inference |
| [runMultimodalHandleWithMessages()](#runmultimodalhandlewithmessages) | Blocking multimodal inference with OpenAI message format |
| [runMultimodalHandleWithMessagesStreaming()](#runmultimodalhandlewithmessagesstreaming) | Streaming multimodal inference with OpenAI message format |

#### runMultimodalHandleStreaming()

```c
ErrorCode runMultimodalHandleStreaming(CausalLmHandle handle,
                                       const char *prompt,
                                       const float *pixelValues,
                                       int numPatches,
                                       int originalHeight,
                                       int originalWidth,
                                       CausalLmTokenCallback callback,
                                       void *user_data);
```

Runs multimodal (image+text) streaming inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Multimodal handle (Vision Encoder + LLM) |
| `prompt` | `const char *` | Text prompt |
| `pixelValues` | `const float *` | Preprocessed image pixel values (CHW format) |
| `numPatches` | `int` | Number of image patches |
| `originalHeight` | `int` | Original image height |
| `originalWidth` | `int` | Original image width |
| `callback` | `CausalLmTokenCallback` | Token callback function |
| `user_data` | `void *` | User data for callback |

**Returns**: `ErrorCode`

**Example (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0;
}

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

// Preprocessed image data (CHW format, 512x512 patch)
float *pixelValues = ...; // Data preprocessed by LlavaNextImageProcessor
int numPatches = 1;
int originalHeight = 512, originalWidth = 512;

runMultimodalHandleStreaming(handle, "Describe this image.", pixelValues,
                             numPatches, originalHeight, originalWidth,
                             token_callback, nullptr);
destroyModelHandle(handle);
```

---

#### runMultimodalHandleWithMessages()

```c
ErrorCode runMultimodalHandleWithMessages(CausalLmHandle handle,
                                          const CausalLMChatMessage *messages,
                                          size_t num_messages,
                                          bool add_generation_prompt,
                                          const float *pixelValues,
                                          int numPatches,
                                          int originalHeight,
                                          int originalWidth,
                                          const char **outputText);
```

Runs multimodal message-based inference (blocking).

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Multimodal handle |
| `messages` | `const CausalLMChatMessage *` | Message array |
| `num_messages` | `size_t` | Number of messages |
| `add_generation_prompt` | `bool` | Whether to add generation prompt |
| `pixelValues` | `const float *` | Preprocessed image pixel values |
| `numPatches` | `int` | Number of image patches |
| `originalHeight` | `int` | Original image height |
| `originalWidth` | `int` | Original image width |
| `outputText` | `const char **` | Pointer to receive output text |

**Returns**: `ErrorCode`

**Example (C++)**:
```cpp
CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

float *pixelValues = ...; // Preprocessed image data
CausalLMChatMessage messages[] = {
    {.role = "user", .content = "What do you see in this image?"}
};
const char *output = nullptr;
ErrorCode err = runMultimodalHandleWithMessages(handle, messages, 1, true,
                                                pixelValues, 1, 512, 512, &output);
if (err == CAUSAL_LM_ERROR_NONE) {
    printf("Output: %s\n", output);
}
destroyModelHandle(handle);
```

---

#### runMultimodalHandleWithMessagesStreaming()

```c
ErrorCode runMultimodalHandleWithMessagesStreaming(CausalLmHandle handle,
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

Runs multimodal message-based streaming inference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Multimodal handle |
| `messages` | `const CausalLMChatMessage *` | Message array |
| `num_messages` | `size_t` | Number of messages |
| `add_generation_prompt` | `bool` | Whether to add generation prompt |
| `pixelValues` | `const float *` | Preprocessed image pixel values |
| `numPatches` | `int` | Number of image patches |
| `originalHeight` | `int` | Original image height |
| `originalWidth` | `int` | Original image width |
| `callback` | `CausalLmTokenCallback` | Token callback function |
| `user_data` | `void *` | User data for callback |

**Returns**: `ErrorCode`

**Example (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0;
}

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

float *pixelValues = ...; // Preprocessed image data
CausalLMChatMessage messages[] = {
    {.role = "user", .content = "Describe what you see."}
};
runMultimodalHandleWithMessagesStreaming(handle, messages, 1, true,
                                         pixelValues, 1, 512, 512,
                                         token_callback, nullptr);
destroyModelHandle(handle);
```

---

### 2.6 XGrammar API - Structured Generation

> **Prerequisite**: XGrammar is integrated for grammar-constrained text generation, ensuring 100% structural correctness of outputs. For detailed usage, see [How to Use XGrammar](../docs/how-to-use-xgrammar.md).

| API | Description |
|---|---|
| [runModelHandleWithTool()](#runmodelhandlewithtool) | Runs inference with grammar-constrained generation for structured output (tools/functions) |

#### runModelHandleWithTool()

```c
ErrorCode runModelHandleWithTool(CausalLmHandle handle,
                                 const char *inputTextPrompt,
                                 const char **outputText,
                                 const char *tool_name,
                                 const char *tool_schema);
```

Runs inference with grammar-constrained generation for structured output.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Model handle |
| `inputTextPrompt` | `const char *` | Input prompt text |
| `outputText` | `const char **` | Pointer to receive output text |
| `tool_name` | `const char *` | Tool name (e.g., "alarm", "send_email") |
| `tool_schema` | `const char *` | JSON schema string (NULL if pre-compiled) |

**Returns**: `ErrorCode`

> **Note**: If `tool_name` exists in pre-compiled `Toolset.json`, `tool_schema` can be NULL. Otherwise, provide a valid JSON schema for dynamic compilation.

**Example (Pre-compiled tool)**:
```cpp
CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

// Tool "alarm" is pre-compiled from Toolset.json
const char *output = nullptr;
ErrorCode err = runModelHandleWithTool(handle, "Set alarm for 7am", 
                                        &output, "alarm", NULL);
if (err == CAUSAL_LM_ERROR_NONE) {
    printf("Output: %s\n", output);
    // Output: {"action": "set", "time": "07:00", "message": "alarm for 7am"}
}
destroyModelHandle(handle);
```

**Example (Dynamic tool registration)**:
```cpp
// Define a new tool schema at runtime
const char* search_schema = R"({
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100}
    },
    "required": ["query"]
})";

const char *output = nullptr;
ErrorCode err = runModelHandleWithTool(handle, "Search for AI news",
                                        &output, "search", search_schema);
// Tool "search" is now registered and can be reused
```

---

## 3. Usage Guide

### 3.1 C/C++ Examples

#### Basic Usage (Blocking)

```cpp
#include "quick_dot_ai_api.h"

int main() {
    // Set options
    Config config = {
        .use_chat_template = true,
        .debug_mode = false,
        .verbose = true,
        .chat_template_name = "default"
    };
    setOptions(config);

    // Load model
    ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU,
                              CAUSAL_LM_MODEL_QWEN3_0_6B,
                              CAUSAL_LM_QUANTIZATION_W4A32,
                              "/path/to/models");
    if (err != CAUSAL_LM_ERROR_NONE) {
        return -1;
    }

    // Build messages
    CausalLMChatMessage messages[] = {
        {.role = "user", .content = "Hello, how are you?"}
    };

    // Run inference
    const char *output = nullptr;
    err = runModelHandleWithMessages(get_default_handle(),
                                     messages, 1, true, &output);
    if (err == CAUSAL_LM_ERROR_NONE) {
        printf("Output: %s\n", output);
    }

    return 0;
}
```

#### Handle-based Usage (Streaming)

```cpp
#include "quick_dot_ai_api.h"
#include <iostream>

int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0; // Continue generation
}

int main() {
    // Create handle and load model
    CausalLmHandle handle = nullptr;
    ErrorCode err = loadModelHandle(CAUSAL_LM_BACKEND_NPU,
                                    CAUSAL_LM_MODEL_GAUSS3_8_QNN,
                                    CAUSAL_LM_QUANTIZATION_W4A32,
                                    "/data/local/tmp/lib",
                                    "/sdcard/models",
                                    &handle);
    if (err != CAUSAL_LM_ERROR_NONE) {
        return -1;
    }

    // Streaming inference
    const char *prompt = "Tell me a story about a brave knight.";
    err = runModelHandleStreaming(handle, prompt, token_callback, nullptr);

    // Get performance metrics
    PerformanceMetrics metrics;
    if (getPerformanceMetricsHandle(handle, &metrics) == CAUSAL_LM_ERROR_NONE) {
        printf("\nGenerated %u tokens in %.2f ms\n",
               metrics.generation_tokens, metrics.generation_duration_ms);
    }

    // Release handle
    destroyModelHandle(handle);
    return 0;
}
```

#### Cancel Inference

```cpp
#include <thread>
#include <atomic>
#include <chrono>

std::atomic<bool> cancel_requested{false};

int cancellable_callback(const char *delta, void *user_data) {
    if (cancel_requested) {
        return 1; // Request cancellation
    }
    std::cout << delta << std::flush;
    return 0;
}

void run_inference(CausalLmHandle handle) {
    runModelHandleStreaming(handle, "Long prompt...", cancellable_callback, nullptr);
}

int main() {
    CausalLmHandle handle = nullptr;
    // ... load model ...

    // Run inference in separate thread
    std::thread inference_thread(run_inference, handle);

    // Cancel after 3 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));
    cancelModelHandle(handle); // or cancel_requested = true

    inference_thread.join();
    destroyModelHandle(handle);
    return 0;
}
```

### 3.2 Kotlin/Java (JNI) Usage Guide

#### JNI Interface Definition

```kotlin
// QuickAI.kt
class QuickAI {
    companion object {
        init {
            System.loadLibrary("quick_dot_ai_api")
        }
    }

    // Native method declarations
    external fun setOptions(useChatTemplate: Boolean, debugMode: Boolean, 
                           verbose: Boolean, templateName: String?): Int

    external fun loadModel(backend: Int, modelType: Int, quantType: Int, 
                          modelBasePath: String?): Int

    external fun loadModelHandle(backend: Int, modelType: Int, quantType: Int,
                                nativeLibDir: String?, modelBasePath: String?): Long

    external fun destroyModelHandle(handle: Long): Int

    external fun runModelWithMessages(handle: Long, messages: Array<Message>,
                                     addGenerationPrompt: Boolean): String?

    external fun cancelModelHandle(handle: Long): Int

    external fun getPerformanceMetrics(handle: Long): PerformanceMetrics?

    data class Message(val role: String, val content: String)
    data class PerformanceMetrics(
        val prefillTokens: Int,
        val prefillDurationMs: Double,
        val generationTokens: Int,
        val generationDurationMs: Double,
        val totalDurationMs: Double,
        val initializationDurationMs: Double,
        val peakMemoryKb: Long
    )
}
```

#### JNI Implementation (C++)

```cpp
// quick_ai_jni.cpp
#include <jni.h>
#include "quick_dot_ai_api.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_QuickAI_loadModelHandle(JNIEnv *env, jobject thiz,
                                         jint backend, jint modelType,
                                         jint quantType, jstring nativeLibDir,
                                         jstring modelBasePath) {
    const char *native_lib = nativeLibDir ? env->GetStringUTFChars(nativeLibDir, nullptr) : nullptr;
    const char *model_path = modelBasePath ? env->GetStringUTFChars(modelBasePath, nullptr) : nullptr;

    CausalLmHandle handle = nullptr;
    ErrorCode err = loadModelHandle((BackendType)backend, (ModelType)modelType,
                                    (ModelQuantizationType)quantType,
                                    native_lib, model_path, &handle);

    if (native_lib) env->ReleaseStringUTFChars(nativeLibDir, native_lib);
    if (model_path) env->ReleaseStringUTFChars(modelBasePath, model_path);

    return (err == CAUSAL_LM_ERROR_NONE) ? (jlong)handle : 0;
}

JNIEXPORT jstring JNICALL
Java_com_example_QuickAI_runModelWithMessages(JNIEnv *env, jobject thiz,
                                              jlong handle, jobjectArray messages,
                                              jboolean addGenerationPrompt) {
    jsize num_messages = env->GetArrayLength(messages);
    std::vector<CausalLMChatMessage> chat_messages(num_messages);

    for (jsize i = 0; i < num_messages; i++) {
        jobject msg_obj = env->GetObjectArrayElement(messages, i);
        jclass msg_class = env->GetObjectClass(msg_obj);

        jstring role = (jstring)env->GetObjectField(msg_obj, 
                        env->GetFieldID(msg_class, "role", "Ljava/lang/String;"));
        jstring content = (jstring)env->GetObjectField(msg_obj,
                          env->GetFieldID(msg_class, "content", "Ljava/lang/String;"));

        chat_messages[i].role = env->GetStringUTFChars(role, nullptr);
        chat_messages[i].content = env->GetStringUTFChars(content, nullptr);

        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
        env->DeleteLocalRef(msg_obj);
        env->DeleteLocalRef(msg_class);
    }

    const char *output = nullptr;
    ErrorCode err = runModelHandleWithMessages((CausalLmHandle)handle,
                                               chat_messages.data(),
                                               num_messages,
                                               addGenerationPrompt,
                                               &output);

    // Cleanup
    for (jsize i = 0; i < num_messages; i++) {
        env->ReleaseStringUTFChars((jstring)nullptr, chat_messages[i].role);
        env->ReleaseStringUTFChars((jstring)nullptr, chat_messages[i].content);
    }

    return (err == CAUSAL_LM_ERROR_NONE && output) ? env->NewStringUTF(output) : nullptr;
}

} // extern "C"
```

#### Kotlin Usage Example

```kotlin
class MainActivity : AppCompatActivity() {
    private var modelHandle: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set options
        QuickAI().setOptions(
            useChatTemplate = true,
            debugMode = false,
            verbose = true,
            templateName = "default"
        )

        // Load model
        val nativeLibDir = applicationInfo.nativeLibraryDir
        val modelPath = getExternalFilesDir(null)?.absolutePath + "/models"

        modelHandle = QuickAI().loadModelHandle(
            backend = 2, // CAUSAL_LM_BACKEND_NPU
            modelType = 3, // CAUSAL_LM_MODEL_GAUSS3_8_QNN
            quantType = 1, // CAUSAL_LM_QUANTIZATION_W4A32
            nativeLibDir = nativeLibDir,
            modelBasePath = modelPath
        )

        // Run inference
        val messages = arrayOf(
            QuickAI.Message("user", "Describe this image.")
        )
        val output = QuickAI().runModelWithMessages(modelHandle, messages, true)
        output?.let { Log.d("QuickAI", "Output: $it") }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (modelHandle != 0L) {
            QuickAI().destroyModelHandle(modelHandle)
        }
    }
}
```

#### Streaming Callback (JNI)

```kotlin
// Callback interface for streaming
interface TokenCallback {
    fun onToken(delta: String): Boolean // true to cancel
}

// JNI streaming wrapper
class StreamingRunner(private val handle: Long) {
    private val callbackRef = AtomicReference<TokenCallback?>(null)

    fun runStreaming(prompt: String, callback: TokenCallback) {
        callbackRef.set(callback)
        nativeRunStreaming(handle, prompt)
    }

    fun cancel() {
        QuickAI().cancelModelHandle(handle)
    }

    private external fun nativeRunStreaming(handle: Long, prompt: String): Int

    // Callback invoked from JNI
    private fun onNativeToken(delta: String): Int {
        val cb = callbackRef.get() ?: return 1
        return if (cb.onToken(delta)) 1 else 0
    }
}
```

---

## 4. Error Code Reference

| Code | Constant | Description | Common Cause |
|------|----------|-------------|--------------|
| 0 | `CAUSAL_LM_ERROR_NONE` | Success | - |
| 1 | `CAUSAL_LM_ERROR_INVALID_PARAMETER` | Invalid parameter | NULL pointer, invalid enum value |
| 2 | `CAUSAL_LM_ERROR_MODEL_LOAD_FAILED` | Model load failed | Missing model file, corrupted model |
| 3 | `CAUSAL_LM_ERROR_INFERENCE_FAILED` | Inference failed | Out of memory, internal error |
| 4 | `CAUSAL_LM_ERROR_NOT_INITIALIZED` | Not initialized | Inference called before model load |
| 5 | `CAUSAL_LM_ERROR_INFERENCE_NOT_RUN` | Inference not run | Metrics queried before inference |
| 6 | `CAUSAL_LM_ERROR_UNSUPPORTED` | Unsupported | Feature not available on this build/handle |
| 99 | `CAUSAL_LM_ERROR_UNKNOWN` | Unknown error | Exception thrown, internal error |

---

## License

Apache License 2.0

Copyright (C) 2026 Samsung Electronics Co., Ltd.
