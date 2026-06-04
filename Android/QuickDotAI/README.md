# QuickDotAI AAR API 📱

`QuickDotAI` is the Android-facing API for Quick.AI. It provides one Kotlin
interface over two engine implementations:

- `NativeQuickDotAI`: JNI path for nntrainer / QNN models, backed by
  `libquickai_jni.so` and the native `quick_dot_ai_api.h` entry points.
- `LiteRTLm`: LiteRT-LM path for Gemma-family `.litertlm` models.

The current Gradle build includes `:QuickDotAI` and `:SampleTestAPP`.

## 📦 Dependency

```kotlin
dependencies {
    implementation(project(":QuickDotAI"))
}
```

Only `arm64-v8a` prebuilt native libraries are currently supported.

## 🧭 API Surface

Package: `com.example.quickdotai`

```kotlin
interface QuickDotAI {
    val kind: String
    val architecture: String?
    val chatSessionId: String?

    fun load(req: LoadModelRequest): BackendResult<Unit>
    fun unload(): BackendResult<Unit>
    fun metrics(): BackendResult<PerformanceMetrics>
    fun cancel()
    fun close()

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

    fun runMultimodalHandle(parts: List<PromptPart>): BackendResult<String>

    fun runMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit>

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

    fun chatRebuild(messages: List<QuickAiChatMessage>): BackendResult<Unit>
    fun chatCancel()
}
```

Removed APIs: `run()`, `runStreaming()`, `runWithMessages()`,
`runWithMessagesStreaming()`, `chatRun()`, and `chatRunStreaming()`.

## 🤖 Engine Selection

Use the `createEngine` factory with a `ModelDescriptor` from `ModelCatalog`. It
picks the engine from the descriptor's `runtime`:

```kotlin
val descriptor = ModelCatalog.byId(ModelIds.GEMMA4) ?: return
val engine: QuickDotAI = createEngine(applicationContext, descriptor)
// RuntimeKind.LITERT -> LiteRTLm, RuntimeKind.NATIVE -> NativeQuickDotAI
```

`gemma4` (LiteRT) is Kotlin-only and never crosses the JNI boundary. Native
model ids are loaded through `loadModelHandleByName()` in `quick_dot_ai_api.h`.

## 💬 OpenAI Message Streaming

Use `runModelHandleWithMessagesStreaming()` for OpenAI-style message lists and
`runModelHandleWithJsonStreaming()` for full OpenAI JSON requests containing
`tools` or legacy `functions`.

End-to-end Chat tab and OpenAI tab examples live in
[`../../docs/ChatAndOpenAIUsage.md`](../../docs/ChatAndOpenAIUsage.md).

## 🖼️ Multimodal Usage

LiteRT-LM multimodal usage requires `LoadModelRequest.visionBackend`.
Native multimodal usage requires a native model handle whose config loads the
expected vision encoder + LLM sub-models.

```kotlin
engine.load(
    LoadModelRequest(
        modelId = ModelIds.GEMMA4,
        backend = BackendType.GPU,
        visionBackend = BackendType.GPU,
        modelPath = "/sdcard/Download/aistudio-mobile/models/gemma-4-E2B-it/gemma-4-E2B-it.litertlm"
    )
)

engine.runMultimodalHandleWithMessagesStreaming(
    listOf(
        QuickAiChatMessage(
            role = QuickAiChatRole.USER,
            parts = listOf(
                PromptPart.ImageFile("/sdcard/photo.jpg"),
                PromptPart.Text("Describe this picture.")
            )
        )
    ),
    sink
)
```

## 🧵 Chat Sessions

Chat sessions keep backend-managed conversation state. Use
`openChatSession()` before `runChatModelHandleStreaming()` or
`runChatMultimodalHandleStreaming()`, then call `chatRebuild()` or
`closeChatSession()` when the conversation state changes or ends. Only one chat
session may be active per engine instance.

See [`../../docs/ChatAndOpenAIUsage.md`](../../docs/ChatAndOpenAIUsage.md) for
complete session examples.

## 🧱 Core Types

```kotlin
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    val modelId: String,
    val quantization: QuantizationType = QuantizationType.W4A32,
    val modelPath: String? = null,
    val visionBackend: BackendType? = null,
    val cacheDir: String? = null,
    val maxNumTokens: Int? = null,
    val nativeLibDir: String? = null,
    val modelBasePath: String? = null,
    val htpBackendConfigPath: String? = null,
)

enum class BackendType { CPU, GPU, NPU }

// Model ids are plain Strings. Well-known ids are exposed as constants
// (see ModelCatalog.kt); the live list comes from ModelCatalog / the native
// catalog, so the AAR is not recompiled when the model list changes.
object ModelIds {
    const val QWEN3_0_6B     = "qwen3-0.6b"
    const val QWEN3_1_7B_Q40 = "qwen3-1.7b-q40"
    const val TINY_BERT      = "tiny-bert"
    const val FUNCTION_GEMMA  = "function-gemma"
    const val GEMMA4         = "gemma4"        // LiteRT only
    const val GEMMA4_CPU     = "gemma4-cpu"
    const val GEMMA4_E2B_QNN = "gemma4-e2b-qnn"
    const val VJEPA_QNN      = "vjepa-qnn"     // V-JEPA multi-image (QNN)
}

enum class QuantizationType { UNKNOWN, W4A32, W16A16, W8A16, W32A32 }

sealed class PromptPart {
    data class Text(val text: String) : PromptPart()
    data class ImageFile(val absolutePath: String) : PromptPart()
    data class ImageBytes(val bytes: ByteArray) : PromptPart()
    // Pre-processed CHW pixel values, used by models like V-JEPA that take
    // externally preprocessed multi-image input.
    data class PreprocessedPixels(
        val pixelValues: FloatArray,
        val numPatches: Int,
        val numImages: Int,
        val patchesPerImage: IntArray,
        val imageHeights: IntArray,
        val imageWidths: IntArray
    ) : PromptPart()
}

data class QuickAiChatMessage(
    val role: QuickAiChatRole,
    val parts: List<PromptPart>
)

enum class QuickAiChatRole { SYSTEM, USER, ASSISTANT }

interface StreamSink {
    fun onDelta(text: String)
    fun onReasoningDelta(text: String) {}
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}
```

See `Types.kt` for the full DTO set, including `QuickAiChatSessionConfig`,
sampling options, error codes, and metrics.

For native QNN models, `htpBackendConfigPath` points to
`htp_backend_ext_config.json`. Absolute paths are used as-is. Relative paths are
resolved from the app external files directory, so
`"configs/htp_backend_ext_config.json"` resolves to
`<externalFilesDir>/configs/htp_backend_ext_config.json`. When omitted,
`NativeQuickDotAI` uses `<externalFilesDir>/htp_backend_ext_config.json`.

## ✅ Rules

- Call `load()` before any inference call.
- Drive each `QuickDotAI` instance from one worker thread.
- Call `close()` when finished; it closes any active chat session.
- Pass `nativeLibDir` for native/QNN models when the host app can provide
  `applicationInfo.nativeLibraryDir`.
- Pass `modelBasePath` for native models when model files live outside the
  native default path.
- Pass `htpBackendConfigPath` for QNN models when
  `htp_backend_ext_config.json` lives outside the app external files root.
- Pass `modelPath` for `LiteRTLm` / `GEMMA4` models.
