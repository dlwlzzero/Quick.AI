# JSON Streaming API 📡

This document describes the current OpenAI JSON streaming path. The historical
file name is kept for compatibility, but the current API name is
`runModelHandleWithJsonStreaming`.

## 🧭 What It Does

`runModelHandleWithJsonStreaming()` accepts an OpenAI-style JSON request,
applies the loaded model's chat template, and streams generated token deltas
through a callback or Android `StreamSink`.

Supported request fields include:

- `messages`
- `tools`
- legacy `functions`
- template options understood by the model's chat template renderer

## 🧠 C API

```c
ErrorCode runModelHandleWithJsonStreaming(CausalLmHandle handle,
                                          const char *jsonRequest,
                                          CausalLmTokenCallback callback,
                                          void *user_data);
```

The model handle must already be loaded, and a chat template must be available
from the model directory. If no template is available, the function returns
`CAUSAL_LM_ERROR_UNSUPPORTED`.

## 📱 Android API

```kotlin
fun runModelHandleWithJsonStreaming(
    jsonRequest: String,
    sink: StreamSink
): BackendResult<Unit>
```

Example:

```kotlin
val request = """
{
  "messages": [
    {"role": "developer", "content": "You can call tools."},
    {"role": "user", "content": "Call mom"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "call",
        "description": "Make a phone call",
        "parameters": {
          "type": "object",
          "properties": {
            "name": { "type": "string" }
          },
          "required": ["name"]
        }
      }
    }
  ]
}
""".trimIndent()

engine.runModelHandleWithJsonStreaming(request, sink)
```

## 📦 Minimal JSON Shapes

Messages only:

```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Write a short joke." }
  ]
}
```

Messages + tools:

```json
{
  "messages": [
    { "role": "developer", "content": "Call the correct tool." },
    { "role": "user", "content": "Remind me tomorrow at 8." }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "reminder",
        "description": "Set a reminder.",
        "parameters": {
          "type": "object",
          "properties": {
            "title": { "type": "string" },
            "start": { "type": "string" }
          },
          "required": ["title", "start"]
        }
      }
    }
  ]
}
```

## 🔄 Implementation Path

| Layer | File | API |
|---|---|---|
| C API | `api/quick_dot_ai_api.h` | `runModelHandleWithJsonStreaming()` |
| C++ implementation | `api/quick_dot_ai_api.cpp` | JSON parse + chat template + streaming |
| JNI | `Android/QuickDotAI/src/main/cpp/quickai_jni.cpp` | native bridge |
| Kotlin JNI | `NativeCausalLm.kt` | `runModelHandleWithJsonStreamingNative()` |
| Kotlin wrapper | `NativeQuickDotAI.kt` | `runModelHandleWithJsonStreaming()` |
| Public interface | `QuickDotAI.kt` | `runModelHandleWithJsonStreaming()` |

## ⚠️ Notes

- `jsonRequest` must be valid UTF-8 JSON.
- `messages` should be present and non-empty for normal chat-template use.
- Tool-call correctness depends on the model and its template.
- This path formats tools/functions through the chat template; for hard
  schema-constrained decoding, use `runModelHandleWithTool()` and XGrammar.

## 📎 Related Docs

- [Chat Templates](ChatTemplate.md)
- [XGrammar Usage](how-to-use-xgrammar.md)
- [C API Reference](../api/README.md)
- [QuickDotAI AAR API](../Android/QuickDotAI/README.md)
