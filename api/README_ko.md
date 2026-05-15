# Quick.AI API 명세서

Quick.AI는 온디바이스 AI 모델 실행을 위한 C API 라이브러리입니다. LLM 추론을 지원하며, 텍스트 생성 및 멀티모달(이미지+텍스트) 처리 기능을 제공합니다.

## 목차

1. [개요](#1-개요)
2. [API 레퍼런스](#2-api-레퍼런스)
3. [사용 가이드](#3-사용-가이드)
4. [오류 코드 참조](#4-오류-코드-참조)

---

## 1. 개요

### 1.1 지원 모델

| 모델 타입 | 모델명 | 설명 |
|-----------|--------|------|
| `CAUSAL_LM_MODEL_QWEN3_0_6B` | QWEN3-0.6B | Qwen3 0.6B 모델 |
| `CAUSAL_LM_MODEL_GAUSS2_5` | GAUSS2.5-1B | Gauss 2.5 1B 모델 |
| `CAUSAL_LM_MODEL_QWEN3_1_7B_Q40` | QWEN3-1.7B-Q40 | Qwen3 1.7B Q40 양자화 모델 |
| `CAUSAL_LM_MODEL_GAUSS3_6` | GAUSS3.6 | Gauss 3.6 모델 |
| `CAUSAL_LM_MODEL_TINY_BERT` | TINY_BERT | Multilingual TinyBERT 모델 |
| `CAUSAL_LM_MODEL_GAUSS3_6_QNN` | GAUSS3.6-QNN | Gauss 3.6 QNN 가속 모델 (QNN 필요) |
| `CAUSAL_LM_MODEL_GAUSS3_8_QNN` | GAUSS3.8-QNN | Gauss 3.8 QNN 가속 모델 (QNN 필요) |
| `CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN` | GAUSS3.8-VE-QNN | Gauss 3.8 Vision Encoder (QNN 필요) |
| `CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN` | GAUSS3.8-VIT-QNN | Gauss 3.8 ViT 모델 (QNN 필요) |

> **Note**: QNN 접미사가 있는 모델은 `ENABLE_QNN` 빌드 옵션 활성화 시에만 사용 가능합니다.

### 1.2 백엔드 타입

| 타입 | 설명 |
|------|------|
| `CAUSAL_LM_BACKEND_CPU` | CPU 백엔드 |
| `CAUSAL_LM_BACKEND_GPU` | GPU 백엔드 (지원 예정)|
| `CAUSAL_LM_BACKEND_NPU` | NPU 백엔드 (QNN 필요) |

### 1.3 양자화 타입

| 타입 | 설명 |
|------|------|
| `CAUSAL_LM_QUANTIZATION_UNKNOWN` | 기본값 |
| `CAUSAL_LM_QUANTIZATION_W4A32` | 4-bit Weight, 32-bit Activation |
| `CAUSAL_LM_QUANTIZATION_W16A16` | 16-bit Weight, 16-bit Activation |
| `CAUSAL_LM_QUANTIZATION_W8A16` | 8-bit Weight, 16-bit Activation |
| `CAUSAL_LM_QUANTIZATION_W32A32` | 32-bit Weight, 32-bit Activation |

---

## 2. API 레퍼런스

### 2.1 설정 및 초기화

#### setOptions()

```c
ErrorCode setOptions(Config config);
```

전역 옵션을 설정합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `config` | `Config` | 설정 구조체 |

**Config 구조체 필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `use_chat_template` | `bool` | 채팅 템플릿 적용 여부 |
| `debug_mode` | `bool` | 디버그 모드 (모델 파일 검증) |
| `verbose` | `bool` | 상세 출력 여부 |
| `chat_template_name` | `const char *` | 템플릿 이름 ("default", "tool_use" 등) |

**반환값**: `ErrorCode`

---

#### registerModelArchitecture()

```c
ErrorCode registerModelArchitecture(const char *arch_name, ModelArchConfig config);
```

새로운 모델 아키텍처를 등록합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `arch_name` | `const char *` | 아키텍처 이름 |
| `config` | `ModelArchConfig` | 아키텍처 설정 |

**반환값**: `ErrorCode`

---

#### registerModel()

```c
ErrorCode registerModel(const char *model_name, const char *arch_name, ModelRuntimeConfig config);
```

새로운 모델을 등록합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `model_name` | `const char *` | 모델 이름 |
| `arch_name` | `const char *` | 아키텍처 이름 |
| `config` | `ModelRuntimeConfig` | 런타임 설정 |

**반환값**: `ErrorCode`

---

### 2.2 레거시 싱글 모델 API

> **Note**: 이 API들은 전역 단일 모델 인스턴스를 사용합니다. 여러 모델을 동시에 실행하려면 핸들 기반 API를 사용하세요.

#### loadModel()

```c
ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type,
                    const char *model_base_path);
```

모델을 로드합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `compute` | `BackendType` | 백엔드 타입 |
| `modeltype` | `ModelType` | 모델 타입 |
| `quant_type` | `ModelQuantizationType` | 양자화 타입 |
| `model_base_path` | `const char *` | 모델 기본 경로 (NULL 시 기본 경로 사용) |

**반환값**: `ErrorCode`

---

#### getPerformanceMetrics()

```c
ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);
```

마지막 추론의 성능 메트릭을 조회합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `metrics` | `PerformanceMetrics *` | 메트릭을 받을 구조체 포인터 |

**반환값**: `ErrorCode`

---

#### applyChatTemplate()

```c
ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                            size_t num_messages,
                            bool add_generation_prompt,
                            const char **formattedText);
```

채팅 메시지에 채팅 템플릿을 적용합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `messages` | `const CausalLMChatMessage *` | 메시지 배열 |
| `num_messages` | `size_t` | 메시지 수 |
| `add_generation_prompt` | `bool` | 생성 프롬프트 추가 여부 |
| `formattedText` | `const char **` | 포맷된 텍스트를 받을 포인터 |

**CausalLMChatMessage 구조체 필드**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `role` | `const char *` | 메시지 역할 ("system", "user", "assistant") |
| `content` | `const char *` | 메시지 내용 |

**반환값**: `ErrorCode`

> **Note**: `formattedText`는 라이브러리 내부 버퍼를 가리키며, 다음 호출 시까지 유효합니다.

---

#### saveQnnKvCache() / loadQnnKvCache() / resetQnnKvCache()

```c
ErrorCode saveQnnKvCache(const char *cache_path);
ErrorCode loadQnnKvCache(const char *cache_path);
ErrorCode resetQnnKvCache(void);
```

QNN KV 캐시를 저장/로드/리셋합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `cache_path` | `const char *` | 캐시 파일 경로 |

**반환값**: `ErrorCode`

> **Note**: QNN 빌드에서만 지원됩니다.

---

### 2.3 핸들 기반 API - 모델 관리

> **Tip**: 핸들 기반 API는 여러 모델을 동시에 로드하고 병렬로 실행할 수 있습니다. 각 핸들은 독립적인 상태를 가집니다.

#### loadModelHandle()

```c
ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          const char *native_lib_dir,
                          const char *model_base_path,
                          CausalLmHandle *out_handle);
```

모델을 로드하고 핸들을 반환합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `compute` | `BackendType` | 백엔드 타입 |
| `modeltype` | `ModelType` | 모델 타입 |
| `quant_type` | `ModelQuantizationType` | 양자화 타입 |
| `native_lib_dir` | `const char *` | 네이티브 라이브러리 경로 (Android용, NULL 가능) |
| `model_base_path` | `const char *` | 모델 기본 경로 |
| `out_handle` | `CausalLmHandle *` | 생성된 핸들을 받을 포인터 |

**반환값**: `ErrorCode`

---

#### destroyModelHandle()

```c
ErrorCode destroyModelHandle(CausalLmHandle handle);
```

핸들과 관련 리소스를 해제합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 해제할 핸들 (NULL 가능) |

**반환값**: `ErrorCode`

> **Note**: NULL 핸들을 전달하면 아무 작업 없이 성공을 반환합니다.

---

#### unloadModelHandle()

```c
ErrorCode unloadModelHandle(CausalLmHandle handle);
```

핸들에서 모델을 언로드합니다 (핸들 구조는 유지).

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |

**반환값**: `ErrorCode`

---

#### getPerformanceMetricsHandle()

```c
ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics);
```

핸들별 성능 메트릭을 조회합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |
| `metrics` | `PerformanceMetrics *` | 메트릭을 받을 구조체 포인터 |

**반환값**: `ErrorCode`

---

#### KV 캐시 관련 (핸들 기반)

```c
ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path);
ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path);
ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle);
```

핸들에 대해 QNN KV 캐시를 저장/로드/리셋합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |
| `cache_path` | `const char *` | 캐시 파일 경로 |

**반환값**: `ErrorCode`

---

### 2.4 핸들 기반 API - 추론 실행 (runModel)

> **Tip**: 추론 실행 함수들은 블로킹 방식과 스트리밍 방식으로 구분됩니다.

#### 스트리밍 API 상세 설명

스트리밍 API는 토큰이 생성될 때마다 콜백 함수를 호출하여 실시간으로 결과를 전달합니다.

**콜백 함수 시그니처**:
```c
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);
```

| 파라미터 | 설명 |
|----------|------|
| `delta` | 현재 토큰에서 생성된 UTF-8 텍스트. 콜백 호출 중에만 유효하며, 필요시 복사해야 함 |
| `user_data` | `runModelHandleStreaming()` 호출 시 전달한 사용자 데이터 포인터 |

**반환값**:
- `0`: 생성 계속
- `0이 아닌 값`: 생성 취소 요청 (다음 토큰 경계에서 중단)

**스트리밍 vs 블로킹 비교**:

| 특성 | 블로킹 API | 스트리밍 API |
|------|-----------|-------------|
| 결과 반환 | 전체 완료 후 한 번 | 토큰마다 실시간 |
| 메모리 | 전체 결과 버퍼 필요 | 토큰 단위 처리 |
| 취소 | `cancelModelHandle()` 호출 필요 | 콜백 반환값으로 즉시 취소 가능 |
| UX | 긴 대기 시간 | 실시간 응답 체감 |
| 용도 | 배치 처리, API 서버 | 채팅 UI, 실시간 생성 |

**스트리밍 사용 시 주의사항**:
1. `delta` 포인터는 콜백 호출 중에만 유효합니다. 데이터를 보관하려면 복사해야 합니다.
2. 콜백 내에서 무거운 작업을 수행하면 생성 속도가 느려질 수 있습니다.
3. 콜백에서 예외를 던지면 안 됩니다 (C API 호환성).
4. 동일 핸들에 대한 동시 호출은 내부적으로 직렬화됩니다.

#### runModelHandleWithMessages()

```c
ErrorCode runModelHandleWithMessages(CausalLmHandle handle,
                                     const CausalLMChatMessage *messages,
                                     size_t num_messages,
                                     bool add_generation_prompt,
                                     const char **outputText);
```

OpenAI 메시지 포맷으로 추론을 실행합니다 (블로킹).

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |
| `messages` | `const CausalLMChatMessage *` | 메시지 배열 |
| `num_messages` | `size_t` | 메시지 수 |
| `add_generation_prompt` | `bool` | 생성 프롬프트 추가 여부 |
| `outputText` | `const char **` | 결과 텍스트를 받을 포인터 |

**반환값**: `ErrorCode`

> **Note**: `outputText`는 핸들이 소유하며, 다음 추론 또는 핸들 파괴 시까지 유효합니다.

**사용 예시 (C++)**:
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

스트리밍 방식으로 추론을 실행합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |
| `inputTextPrompt` | `const char *` | 입력 프롬프트 (UTF-8) |
| `callback` | `CausalLmTokenCallback` | 토큰 콜백 함수 |
| `user_data` | `void *` | 콜백에 전달할 사용자 데이터 |

**반환값**: `ErrorCode`

**사용 예시 (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0; // 0: 계속 생성, 0이 아니면 취소
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

실행 중인 추론을 취소합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |

**반환값**: `ErrorCode`

> **Note**: 스레드 안전하며, 어떤 스레드에서도 호출 가능합니다.

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

OpenAI 메시지 포맷으로 스트리밍 추론을 실행합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 모델 핸들 |
| `messages` | `const CausalLMChatMessage *` | 메시지 배열 |
| `num_messages` | `size_t` | 메시지 수 |
| `add_generation_prompt` | `bool` | 생성 프롬프트 추가 여부 |
| `callback` | `CausalLmTokenCallback` | 토큰 콜백 함수 |
| `user_data` | `void *` | 콜백에 전달할 사용자 데이터 |

**반환값**: `ErrorCode`

**사용 예시 (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0; // 0: 계속 생성
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

### 2.5 멀티모달 API

> **Prerequisite**: 멀티모달 API는 QNN 빌드에서만 지원됩니다. 핸들이 Vision Encoder와 LLM 두 개의 서브 모델로 로드되어야 합니다.

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

멀티모달(이미지+텍스트) 스트리밍 추론을 실행합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 멀티모달 핸들 (Vision Encoder + LLM) |
| `prompt` | `const char *` | 텍스트 프롬프트 |
| `pixelValues` | `const float *` | 전처리된 이미지 픽셀 값 (CHW 포맷) |
| `numPatches` | `int` | 이미지 패치 수 |
| `originalHeight` | `int` | 원본 이미지 높이 |
| `originalWidth` | `int` | 원본 이미지 너비 |
| `callback` | `CausalLmTokenCallback` | 토큰 콜백 함수 |
| `user_data` | `void *` | 콜백 사용자 데이터 |

**반환값**: `ErrorCode`

**사용 예시 (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0;
}

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

// 전처리된 이미지 데이터 (CHW 포맷, 512x512 패치)
float *pixelValues = ...; // LlavaNextImageProcessor에서 전처리된 데이터
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

멀티모달 메시지 기반 추론을 실행합니다 (블로킹).

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 멀티모달 핸들 |
| `messages` | `const CausalLMChatMessage *` | 메시지 배열 |
| `num_messages` | `size_t` | 메시지 수 |
| `add_generation_prompt` | `bool` | 생성 프롬프트 추가 여부 |
| `pixelValues` | `const float *` | 전처리된 이미지 픽셀 값 |
| `numPatches` | `int` | 이미지 패치 수 |
| `originalHeight` | `int` | 원본 이미지 높이 |
| `originalWidth` | `int` | 원본 이미지 너비 |
| `outputText` | `const char **` | 결과 텍스트를 받을 포인터 |

**반환값**: `ErrorCode`

**사용 예시 (C++)**:
```cpp
CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

float *pixelValues = ...; // 전처리된 이미지 데이터
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

멀티모달 메시지 기반 스트리밍 추론을 실행합니다.

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `handle` | `CausalLmHandle` | 멀티모달 핸들 |
| `messages` | `const CausalLMChatMessage *` | 메시지 배열 |
| `num_messages` | `size_t` | 메시지 수 |
| `add_generation_prompt` | `bool` | 생성 프롬프트 추가 여부 |
| `pixelValues` | `const float *` | 전처리된 이미지 픽셀 값 |
| `numPatches` | `int` | 이미지 패치 수 |
| `originalHeight` | `int` | 원본 이미지 높이 |
| `originalWidth` | `int` | 원본 이미지 너비 |
| `callback` | `CausalLmTokenCallback` | 토큰 콜백 함수 |
| `user_data` | `void *` | 콜백 사용자 데이터 |

**반환값**: `ErrorCode`

**사용 예시 (C++)**:
```cpp
int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0;
}

CausalLmHandle handle = nullptr;
loadModelHandle(CAUSAL_LM_BACKEND_NPU, CAUSAL_LM_MODEL_GAUSS3_8_VE_QNN,
                CAUSAL_LM_QUANTIZATION_W4A32, nullptr, "/models", &handle);

float *pixelValues = ...; // 전처리된 이미지 데이터
CausalLMChatMessage messages[] = {
    {.role = "user", .content = "Describe what you see."}
};
runMultimodalHandleWithMessagesStreaming(handle, messages, 1, true,
                                         pixelValues, 1, 512, 512,
                                         token_callback, nullptr);
destroyModelHandle(handle);
```

---

## 3. 사용 가이드

### 3.1 C/C++ 사용 예제

#### 기본 사용 (블로킹)

```cpp
#include "quick_dot_ai_api.h"

int main() {
    // 옵션 설정
    Config config = {
        .use_chat_template = true,
        .debug_mode = false,
        .verbose = true,
        .chat_template_name = "default"
    };
    setOptions(config);

    // 모델 로드
    ErrorCode err = loadModel(CAUSAL_LM_BACKEND_CPU,
                              CAUSAL_LM_MODEL_QWEN3_0_6B,
                              CAUSAL_LM_QUANTIZATION_W4A32,
                              "/path/to/models");
    if (err != CAUSAL_LM_ERROR_NONE) {
        return -1;
    }

    // 메시지 구성
    CausalLMChatMessage messages[] = {
        {.role = "user", .content = "Hello, how are you?"}
    };

    // 추론 실행
    const char *output = nullptr;
    err = runModelHandleWithMessages(get_default_handle(),
                                     messages, 1, true, &output);
    if (err == CAUSAL_LM_ERROR_NONE) {
        printf("Output: %s\n", output);
    }

    return 0;
}
```

#### 핸들 기반 사용 (스트리밍)

```cpp
#include "quick_dot_ai_api.h"
#include <iostream>

int token_callback(const char *delta, void *user_data) {
    std::cout << delta << std::flush;
    return 0; // 계속 생성
}

int main() {
    // 핸들 생성 및 모델 로드
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

    // 스트리밍 추론
    const char *prompt = "Tell me a story about a brave knight.";
    err = runModelHandleStreaming(handle, prompt, token_callback, nullptr);

    // 성능 메트릭 조회
    PerformanceMetrics metrics;
    if (getPerformanceMetricsHandle(handle, &metrics) == CAUSAL_LM_ERROR_NONE) {
        printf("\nGenerated %u tokens in %.2f ms\n",
               metrics.generation_tokens, metrics.generation_duration_ms);
    }

    // 핸들 해제
    destroyModelHandle(handle);
    return 0;
}
```

#### 추론 취소

```cpp
#include <thread>
#include <atomic>
#include <chrono>

std::atomic<bool> cancel_requested{false};

int cancellable_callback(const char *delta, void *user_data) {
    if (cancel_requested) {
        return 1; // 취소 요청
    }
    std::cout << delta << std::flush;
    return 0;
}

void run_inference(CausalLmHandle handle) {
    runModelHandleStreaming(handle, "Long prompt...", cancellable_callback, nullptr);
}

int main() {
    CausalLmHandle handle = nullptr;
    // ... 모델 로드 ...

    // 별도 스레드에서 추론 실행
    std::thread inference_thread(run_inference, handle);

    // 3초 후 취소
    std::this_thread::sleep_for(std::chrono::seconds(3));
    cancelModelHandle(handle); // 또는 cancel_requested = true

    inference_thread.join();
    destroyModelHandle(handle);
    return 0;
}
```

### 3.2 Kotlin/Java (JNI) 사용 가이드

#### JNI 인터페이스 정의

```kotlin
// QuickAI.kt
class QuickAI {
    companion object {
        init {
            System.loadLibrary("quick_dot_ai_api")
        }
    }

    // 네이티브 메서드 선언
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

#### JNI 구현 (C++)

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

    // 정리
    for (jsize i = 0; i < num_messages; i++) {
        env->ReleaseStringUTFChars((jstring)nullptr, chat_messages[i].role);
        env->ReleaseStringUTFChars((jstring)nullptr, chat_messages[i].content);
    }

    return (err == CAUSAL_LM_ERROR_NONE && output) ? env->NewStringUTF(output) : nullptr;
}

} // extern "C"
```

#### Kotlin 사용 예제

```kotlin
class MainActivity : AppCompatActivity() {
    private var modelHandle: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 옵션 설정
        QuickAI().setOptions(
            useChatTemplate = true,
            debugMode = false,
            verbose = true,
            templateName = "default"
        )

        // 모델 로드
        val nativeLibDir = applicationInfo.nativeLibraryDir
        val modelPath = getExternalFilesDir(null)?.absolutePath + "/models"

        modelHandle = QuickAI().loadModelHandle(
            backend = 2, // CAUSAL_LM_BACKEND_NPU
            modelType = 3, // CAUSAL_LM_MODEL_GAUSS3_8_QNN
            quantType = 1, // CAUSAL_LM_QUANTIZATION_W4A32
            nativeLibDir = nativeLibDir,
            modelBasePath = modelPath
        )

        // 추론 실행
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

#### 스트리밍 콜백 (JNI)

```kotlin
// 스트리밍을 위한 콜백 인터페이스
interface TokenCallback {
    fun onToken(delta: String): Boolean // true면 취소
}

// JNI 스트리밍 래퍼
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

    // JNI에서 호출하는 콜백
    private fun onNativeToken(delta: String): Int {
        val cb = callbackRef.get() ?: return 1
        return if (cb.onToken(delta)) 1 else 0
    }
}
```

---

## 4. 오류 코드 참조

| 코드 | 상수 | 설명 | 일반적 원인 |
|------|------|------|------------|
| 0 | `CAUSAL_LM_ERROR_NONE` | 성공 | - |
| 1 | `CAUSAL_LM_ERROR_INVALID_PARAMETER` | 잘못된 파라미터 | NULL 포인터, 잘못된 enum 값 |
| 2 | `CAUSAL_LM_ERROR_MODEL_LOAD_FAILED` | 모델 로드 실패 | 모델 파일 없음, 손상된 모델 |
| 3 | `CAUSAL_LM_ERROR_INFERENCE_FAILED` | 추론 실패 | 메모리 부족, 내부 오류 |
| 4 | `CAUSAL_LM_ERROR_NOT_INITIALIZED` | 초기화되지 않음 | 모델 로드 전 추론 호출 |
| 5 | `CAUSAL_LM_ERROR_INFERENCE_NOT_RUN` | 추론 실행 안 됨 | 메트릭 조회 전 추론 필요 |
| 6 | `CAUSAL_LM_ERROR_UNSUPPORTED` | 지원하지 않음 | QNN 없이 QNN 기능 사용 |
| 99 | `CAUSAL_LM_ERROR_UNKNOWN` | 알 수 없는 오류 | 예외 발생, 내부 오류 |

---

## 라이선스

Apache License 2.0

Copyright (C) 2026 Samsung Electronics Co., Ltd.
