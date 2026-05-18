# QuickDotAI AAR — API

On-device LLM inference. `NativeQuickDotAI` routes non-Gemma models
through JNI to `libcausallm_api.so`; `LiteRTLm` routes Gemma-family
models through LiteRT-LM and also supports image input.

## Dependency

```kotlin
dependencies {
    implementation(project(":QuickDotAI"))
}
```

## API surface (`com.example.quickdotai`)

```kotlin
interface QuickDotAI {
    val kind: String                       // "native" or "litert-lm"
    val architecture: String?
    val chatSessionId: String?             // null if no session active

    // Lifecycle
    fun load(req: LoadModelRequest): BackendResult<Unit>
    fun unload(): BackendResult<Unit>
    fun metrics(): BackendResult<PerformanceMetrics>
    fun close()
    fun cancel()

    // Handle-based OpenAI messages API (OpenAI Tab)
    fun runModelHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit>

    fun runMultimodalHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit>

    fun runModelHandleWithJsonStreaming(
        jsonRequest: String,
        sink: StreamSink
    ): BackendResult<Unit>

    // Handle-based multimodal parts API (internal/chat image)
    fun runMultimodalHandle(
        parts: List<PromptPart>
    ): BackendResult<String>

    fun runMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit>

    // Chat session API (Chat Tab)
    fun openChatSession(
        config: QuickAiChatSessionConfig? = null
    ): BackendResult<String>

    fun closeChatSession(): BackendResult<Unit>

    fun runChatModelHandleStreaming(
        text: String,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult>

    fun runChatMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult>

    fun chatRebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit>

    fun chatCancel()
}
```

## Tab/Model Usage Matrix

### OpenAI Tab — Handle-based Messages

```kotlin
// Text-only models (GAUSS3_8_QNN, GAUSS3_6_QNN, QWEN3, etc.)
val messages = listOf(
    QuickAiChatMessage(role = QuickAiChatRole.SYSTEM, 
        parts = listOf(PromptPart.Text("You are a helpful assistant."))),
    QuickAiChatMessage(role = QuickAiChatRole.USER, 
        parts = listOf(PromptPart.Text("Hello!")))
)
engine.runModelHandleWithMessagesStreaming(messages, sink)

// Vision models with image (GEMMA4, GAUSS3_8_VISION_QNN)
val imageBytes = loadImageBytes("/sdcard/photo.jpg")
val messages = listOf(
    QuickAiChatMessage(role = QuickAiChatRole.USER, parts = listOf(
        PromptPart.Text("Describe this image."),
        PromptPart.ImageBytes(imageBytes)
    ))
)
engine.runMultimodalHandleWithMessagesStreaming(messages, sink)

// Full OpenAI JSON (tools, functions, etc.)
val jsonRequest = """
{
  "messages": [
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "..."}}
  ]
}
""".trimIndent()
engine.runModelHandleWithJsonStreaming(jsonRequest, sink)
```

### Chat Tab — Session-based

```kotlin
// Open chat session
engine.openChatSession()

// Text-only chat
text = "Hello!"
engine.runChatModelHandleStreaming(text, sink)

// Chat with image
val parts = listOf(
    PromptPart.ImageBytes(imageBytes),
    PromptPart.Text("Describe this image.")
)
engine.runChatMultimodalHandleStreaming(parts, sink)

// Close session
engine.closeChatSession()
```

## Data Types

```kotlin
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    val model: ModelId,
    val quantization: QuantizationType = QuantizationType.W4A32,
    val modelPath: String? = null,         // required for GEMMA4
    val visionBackend: BackendType? = null, // non-null enables multimodal
    val cacheDir: String? = null,
    val nativeLibDir: File? = null,        // for QNN .so loading
)

sealed class PromptPart {
    data class Text(val text: String) : PromptPart()
    data class ImageFile(val absolutePath: String) : PromptPart()
    data class ImageBytes(val bytes: ByteArray) : PromptPart()
}

data class QuickAiChatMessage(
    val role: QuickAiChatRole,
    val parts: List<PromptPart>
)

enum class QuickAiChatRole {
    SYSTEM, USER, ASSISTANT
}

sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(val error: QuickAiError, val message: String? = null) : BackendResult<Nothing>()
}

interface StreamSink {
    fun onDelta(text: String)
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}

enum class BackendType      { CPU, GPU, NPU }
enum class ModelId          { QWEN3_0_6B, GAUSS2_5, GAUSS3_6, GAUSS3_8, GEMMA4, GEMMA4_CPU, GAUSS3_6_QNN, GAUSS3_8_QNN, GAUSS3_8_VISION_QNN }
enum class QuantizationType { UNKNOWN, W4A32, W16A16, W8A16, W32A32 }
enum class QuickAiError {
    NONE, INVALID_PARAMETER, MODEL_LOAD_FAILED, INFERENCE_FAILED,
    NOT_INITIALIZED, INFERENCE_NOT_RUN, UNKNOWN,
    QUEUE_FULL, MODEL_NOT_FOUND, UNSUPPORTED, BAD_REQUEST
}

data class PerformanceMetrics(
    val prefillTokens: Int, val prefillDurationMs: Double,
    val generationTokens: Int, val generationDurationMs: Double,
    val totalDurationMs: Double, val initializationDurationMs: Double,
    val peakMemoryKb: Long,
)
```

## Minimal example

```kotlin
val engine: QuickDotAI = when (req.model) {
    ModelId.GEMMA4 -> LiteRTLm(applicationContext)
    else           -> NativeQuickDotAI(applicationContext)
}

engine.load(LoadModelRequest(
    model = ModelId.GEMMA4,
    backend = BackendType.GPU,
    visionBackend = BackendType.GPU,     // enables images
    modelPath = "/sdcard/.../gemma-4-E2B-it.litertlm",
))

// OpenAI-style messages with image
val messages = listOf(
    QuickAiChatMessage(role = QuickAiChatRole.USER, parts = listOf(
        PromptPart.ImageFile("/sdcard/photo.jpg"),
        PromptPart.Text("Describe this picture."),
    ))
)
engine.runMultimodalHandleWithMessagesStreaming(messages, sink)

engine.close()
```

## Rules

- Call `load()` exactly once before any inference.
- A single instance is **not thread-safe** — drive it from one worker thread.
- `runMultimodalHandle*` on a text-only engine (or `NativeQuickDotAI` without visionBackend) returns `QuickAiError.UNSUPPORTED`.
- `arm64-v8a` only.
- **Removed APIs** (no longer available): `run()`, `runStreaming()`, `runWithMessages()`, `runWithMessagesStreaming()`, `chatRun()`, `chatRunStreaming()`. Use the new Handle/Chat streaming APIs instead.
