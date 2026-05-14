# runWithJsonStreaming API

OpenAI JSON format의 tool/function 호출을 지원하는 streaming API입니다.

## 개요

`runWithJsonStreaming`은 OpenAI format의 JSON 문자열을 직접 받아서 minja chat template을 통해 처리하는 API입니다. `messages`, `tools`, `functions` 등 모든 OpenAI format 필드를 지원합니다.

## API 시그니처

```kotlin
fun runWithJsonStreaming(
    jsonRequest: String,
    sink: StreamSink
): BackendResult<Unit>
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `jsonRequest` | String | OpenAI format JSON 문자열 |
| `sink` | StreamSink | 스트리밍 출력을 받을 콜백 인터페이스 |

### 반환값

- `BackendResult.Ok(Unit)`: 성공
- `BackendResult.Err(error, message)`: 실패

## JSON 입력 형식

### 기본 형식 (messages만)

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Write a short joke."}
    ]
}
```

### Tool Calling 형식 (messages + tools)

```json
{
    "messages": [
        {
            "role": "developer",
            "content": "Analyze the given <conversation> and call the appropriate tool. If no tool applies, call `unknown`.\n\nAvailable tools:\n- call: Make a call or send a text. Params: name (contact name), number (phone number)\n- reminder: Set a reminder. Params: title (reminder title), start (reminder time)\n- unknown: No actionable tool applies. Params: (none)"
        },
        {
            "role": "user",
            "content": "Call mom at 555-1234"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "call",
                "description": "Make a call or send a text to a contact.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {},
                        "number": {}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reminder",
                "description": "Set a reminder.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {},
                        "start": {}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "unknown",
                "description": "No actionable tool call applies.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {}
                }
            }
        }
    ]
}
```

## 사용 예시

### Kotlin (Android)

```kotlin
// 1. QuickDotAI 인스턴스 생성 및 모델 로드
val engine = NativeQuickDotAI(context)
engine.load(LoadModelRequest(
    model = ModelId.FUNCTION_GEMMA,
    backend = BackendType.CPU,
    modelBasePath = "/sdcard/Android/data/com.example.yourapp/files/models"
))

// 2. JSON 요청 준비
val jsonRequest = """
{
    "messages": [
        {"role": "developer", "content": "Analyze and call appropriate tool..."},
        {"role": "user", "content": "Call mom at 555-1234"}
    ],
    "tools": [
        {"type": "function", "function": {"name": "call", "description": "Make a call", "parameters": {"type": "OBJECT", "properties": {"name": {}, "number": {}}}}}
    ]
}
"""

// 3. Streaming 호출
val result = engine.runWithJsonStreaming(jsonRequest, object : StreamSink {
    override fun onDelta(text: String) {
        // 각 토큰 수신
        println("Token: $text")
    }
    override fun onDone() {
        println("완료!")
    }
    override fun onError(error: QuickAiError, message: String?) {
        println("에러: $error - $message")
    }
})

// 4. 완료 후 리소스 해제
engine.close()
```

## 구현 계층

| 계층 | 파일 | 함수 |
|------|------|------|
| C API | `api/quick_dot_ai_api.h` | `runModelHandleWithJsonStreaming()` |
| C++ 구현 | `api/quick_dot_ai_api.cpp` | JSON 파싱 및 chat_template 연동 |
| JNI | `Android/QuickDotAI/src/main/cpp/quickai_jni.cpp` | `runModelHandleWithJsonStreamingNative()` |
| Kotlin JNI | `NativeCausalLm.kt` | JNI 함수 선언 |
| Kotlin 구현 | `NativeQuickDotAI.kt` | `runWithJsonStreaming()` 구현 |
| Kotlin 인터페이스 | `QuickDotAI.kt` | `runWithJsonStreaming()` 인터페이스 |

## 내부 동작

1. **JSON 파싱**: 입력된 JSON 문자열을 nlohmann::json으로 파싱
2. **Chat Template 적용**: minja chat template을 사용하여 messages/tools를 포맷팅
3. **토큰화**: 포맷팅된 텍스트를 토큰화
4. **추론**: 스트리밍 방식으로 토큰 생성
5. **콜백**: 각 토큰마다 `onDelta()` 호출

## tools 유무에 따른 동작

| 입력 | 동작 |
|------|------|
| `messages`만 | 일반 chat 응답 (텍스트) |
| `messages` + `tools` | Tool calling 형식 응답 (`{"name": "...", "arguments": {...}}`) |

## 지원 모델

- `FUNCTION_GEMMA`: Tool calling에 최적화된 모델
- 기타 CausalLM 기반 모델들 (tool calling 지원 여부는 모델에 따라 다름)

## 주의사항

1. JSON 문자열은 유효한 JSON 형식이어야 합니다.
2. `messages` 배열은 필수이며, 최소 하나의 `user` role 메시지가 있어야 합니다.
3. `tools`는 선택사항입니다.
4. 스레드 안전성: 동일한 handle에 대한 동시 호출은 지원되지 않습니다.

## 관련 파일

- 입력 예시: `function_gemma_input.txt`
- Chat Template 문서: `ChatTemplate.md`
