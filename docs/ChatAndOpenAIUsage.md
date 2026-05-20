# Chat and OpenAI Usage Examples 💬

This is the canonical place for Quick.AI chat, OpenAI-style request, and
structured-output examples. Keep API signatures in the API reference docs; keep
end-to-end usage examples here.

## 🧭 Choose the Right API

| Goal | Android API | Native C API | Notes |
|---|---|---|---|
| Chat tab style conversation | `openChatSession()` → `runChatModelHandleStreaming()` → `closeChatSession()` | session is Android-facing | Keeps backend-managed conversation state. |
| OpenAI-style messages | `runModelHandleWithMessagesStreaming()` | `runModelHandleWithMessagesStreaming()` | Best for models that need message-template formatting but not full JSON fields. |
| Full OpenAI JSON request | `runModelHandleWithJsonStreaming()` | `runModelHandleWithJsonStreaming()` | Preserves `messages`, `tools`, legacy `functions`, and template kwargs understood by the chat template. |
| Hard schema-constrained output | not exposed by the AAR today | `runModelHandleWithTool()` | Uses XGrammar token masking. This is different from OpenAI JSON `tools`. |

`tools` in an OpenAI JSON request are passed into the model's chat template.
They help the model see tool metadata and produce a tool-call shaped answer.
XGrammar is stricter: it masks invalid tokens during decoding so the output
matches a JSON schema.

## ⚙️ Common Setup

Every path needs a loaded engine or handle before inference.

```kotlin
val engine: QuickDotAI = when (request.model) {
    ModelId.GEMMA4 -> LiteRTLm(applicationContext)
    else -> NativeQuickDotAI(applicationContext)
}

val loaded = engine.load(
    LoadModelRequest(
        model = ModelId.GAUSS3_8_QNN,
        backend = BackendType.NPU,
        quantization = QuantizationType.W4A32,
        nativeLibDir = applicationInfo.nativeLibraryDir,
        modelBasePath = "/sdcard/Download/aistudio-mobile/models"
    )
)
```

Streaming methods report deltas through `StreamSink`.

```kotlin
val sink = object : StreamSink {
    override fun onDelta(text: String) {
        outputView.append(text)
    }

    override fun onDone() {
        setStatus("Done.")
    }

    override fun onError(error: QuickAiError, message: String?) {
        setStatus("Failed: [${error.name}] ${message.orEmpty()}")
    }
}
```

Message and JSON APIs require a usable chat template. The native loader checks
the model directory for `chat_template.jinja` or
`tokenizer_config.json.chat_template`; see [Chat Templates](ChatTemplate.md).

## 💬 Chat Tab Pattern

The Chat tab is session based. It opens one active session per engine, sends
turns into that session, and closes it when the user leaves or reloads.

```kotlin
val config = QuickAiChatSessionConfig(
    systemInstruction = "You are concise.",
    sampling = QuickAiChatSamplingConfig(
        temperature = 0.7,
        topK = 40,
        topP = 0.9,
        seed = 42
    ),
    chatTemplateKwargs = QuickAiChatTemplateKwargs(enableThinking = false)
)

when (val opened = engine.openChatSession(config)) {
    is BackendResult.Ok -> {
        val sessionId = opened.value
        setStatus("Chat session opened: ${sessionId.take(8)}")
    }
    is BackendResult.Err -> {
        setStatus("Chat open failed: [${opened.error.name}] ${opened.message.orEmpty()}")
    }
}
```

Send text through the active session:

```kotlin
when (val result = engine.runChatModelHandleStreaming("Explain KV cache.", sink)) {
    is BackendResult.Ok -> {
        val metrics = result.value.metrics
        setStatus("Chat done. ${metrics?.totalDurationMs?.toLong() ?: "?"} ms")
    }
    is BackendResult.Err -> {
        setStatus("Chat failed: [${result.error.name}] ${result.message.orEmpty()}")
    }
}
```

Reset or close the session when needed:

```kotlin
engine.chatRebuild(emptyList())
engine.closeChatSession()
```

For multimodal chat input, build `PromptPart` values. The sample app currently
routes image-attached chat turns through the direct multimodal handle path.

```kotlin
val parts = listOf(
    PromptPart.ImageBytes(imageBytes),
    PromptPart.Text("Describe the image.")
)

engine.runMultimodalHandleStreaming(parts, sink)
```

## 📡 OpenAI Tab Pattern

The OpenAI tab accepts OpenAI-style JSON from the UI and then routes by model.
Gauss models and `GEMMA4` use the messages API because they need model-specific
message formatting. Other native models can use full JSON streaming.

```kotlin
private val messagesApiModels = setOf(
    ModelId.GAUSS3_6_QNN,
    ModelId.GAUSS3_8_QNN,
    ModelId.GAUSS3_8_VISION_QNN,
    ModelId.GAUSS3_8,
    ModelId.GAUSS3_6,
    ModelId.GEMMA4
)

if (selectedModel in messagesApiModels) {
    val messages = parseOpenAIMessages(jsonText) ?: return
    engine.runModelHandleWithMessagesStreaming(messages, sink)
} else {
    engine.runModelHandleWithJsonStreaming(jsonText, sink)
}
```

Use messages streaming when the request is just an ordered chat history:

```kotlin
val messages = listOf(
    QuickAiChatMessage(
        role = QuickAiChatRole.SYSTEM,
        parts = listOf(PromptPart.Text("You are concise."))
    ),
    QuickAiChatMessage(
        role = QuickAiChatRole.USER,
        parts = listOf(PromptPart.Text("Write a one-line haiku."))
    )
)

engine.runModelHandleWithMessagesStreaming(messages, sink)
```

Use JSON streaming when you need the full OpenAI request shape:

```kotlin
val jsonRequest = """
{
  "messages": [
    {"role": "developer", "content": "You can call tools."},
    {"role": "user", "content": "Set an alarm for 7 AM."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "set_alarm",
        "description": "Create an alarm.",
        "parameters": {
          "type": "object",
          "properties": {
            "time": {"type": "string"},
            "label": {"type": "string"}
          },
          "required": ["time"]
        }
      }
    }
  ]
}
""".trimIndent()

engine.runModelHandleWithJsonStreaming(jsonRequest, sink)
```

Legacy OpenAI `functions` are accepted by the chat-template renderer as raw
function schemas:

```json
{
  "messages": [
    {"role": "user", "content": "Send a short status email."}
  ],
  "functions": [
    {
      "name": "send_email",
      "description": "Send email.",
      "parameters": {
        "type": "object",
        "properties": {
          "to": {"type": "string"},
          "body": {"type": "string"}
        },
        "required": ["to", "body"]
      }
    }
  ]
}
```

## 🧱 Native Call Flow

The Android wrapper is thin. It converts Kotlin DTOs to native inputs and then
uses the C API.

| User path | JNI/native path | Formatting step | Inference step |
|---|---|---|---|
| `runModelHandleWithMessagesStreaming()` | `QuickAiChatMessage[]` → `CausalLMChatMessage[]` | `apply_chat_template_messages()` | `run_model_streaming_on_handle(..., input_already_formatted=true)` |
| `runModelHandleWithJsonStreaming()` | JSON string | `g_chat_template->apply(request)` | `run_model_streaming_on_handle(..., input_already_formatted=true)` |
| `runModelHandleWithTool()` | prompt + tool name/schema | `XGrammarManager` attaches grammar | `run_on_handle()` with grammar mask active |

Native messages example:

```c
CausalLMChatMessage messages[] = {
  {.role = "system", .content = "You are concise."},
  {.role = "user", .content = "Hello!"}
};

ErrorCode err = runModelHandleWithMessagesStreaming(
    handle,
    messages,
    2,
    true,
    callback,
    user_data);
```

Native JSON streaming example:

```c
const char *request =
    "{"
    "\"messages\":["
    "{\"role\":\"user\",\"content\":\"Summarize Quick.AI.\"}"
    "]"
    "}";

ErrorCode err = runModelHandleWithJsonStreaming(
    handle,
    request,
    callback,
    user_data);
```

## 🧩 XGrammar Examples

Use XGrammar when you need the generated text itself to obey a JSON schema. A
model directory can preload grammars from `Toolset.json`.

```json
{
  "set_alarm": {
    "type": "object",
    "properties": {
      "time": {"type": "string", "pattern": "^\\d{2}:\\d{2}$"},
      "label": {"type": "string"}
    },
    "required": ["time"]
  }
}
```

After `loadModelHandle()`, call the tool path by name:

```c
const char *output = NULL;

ErrorCode err = runModelHandleWithTool(
    handle,
    "Create an alarm for 07:00.",
    &output,
    "set_alarm",
    NULL);
```

For an ad-hoc schema, pass the schema on first use:

```c
const char *schema =
    "{"
    "\"type\":\"object\","
    "\"properties\":{\"answer\":{\"type\":\"string\"}},"
    "\"required\":[\"answer\"]"
    "}";

ErrorCode err = runModelHandleWithTool(
    handle,
    "Return the answer as JSON.",
    &output,
    "answer_schema",
    schema);
```

The XGrammar manager compiles schemas once and can load/save
`Toolset.json.cache` for faster subsequent loads. See
[XGrammar Reference](XGrammarReference.md) for cache behavior and C++ manager
details.

## ✅ Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `runModelHandleWithJsonStreaming()` returns `CAUSAL_LM_ERROR_UNSUPPORTED` | The loaded model has no chat template cached. | Add `chat_template.jinja` or `tokenizer_config.json.chat_template` next to the model config. |
| JSON streaming returns `CAUSAL_LM_ERROR_INVALID_PARAMETER` | The request is not valid JSON or a required pointer is null. | Validate the JSON and ensure `messages` is non-empty for normal chat use. |
| OpenAI tab loses `tools` on Gauss/GEMMA4 models | The sample routes those models through messages streaming. | Use a model that supports full JSON streaming, or add a dedicated model-specific full JSON path. |
| `tools` are visible to the model but output is not schema-valid | OpenAI JSON `tools` only guide the chat template. | Use `runModelHandleWithTool()` and XGrammar for hard constraints. |
| Chat tab says no active session | `openChatSession()` has not succeeded or the session was closed. | Open a session first, then call `runChatModelHandleStreaming()`. |

## 📎 Related Docs

- [QuickDotAI AAR API](../Android/QuickDotAI/README.md)
- [C API Reference](../api/README.md)
- [Chat Templates](ChatTemplate.md)
- [XGrammar Reference](XGrammarReference.md)
