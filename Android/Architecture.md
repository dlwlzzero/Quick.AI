# Android Architecture 📱

This document describes the current Android state of Quick.AI and separates it
from the planned REST/foreground-service layer that older documents described
as if it already existed.

## ✅ Current Gradle Modules

The Android build currently includes:

```text
Android/
├── QuickDotAI/       # AAR module
└── SampleTestAPP/    # Direct sample app using the AAR
```

`Android/settings.gradle.kts` includes only `:QuickDotAI` and
`:SampleTestAPP`.

## 🧱 QuickDotAI AAR

`QuickDotAI` exposes the public Kotlin API in
`com.example.quickdotai`.

Key files:

| File | Role |
|---|---|
| `QuickDotAI.kt` | Public interface and `BackendResult` / `StreamSink` contracts |
| `Types.kt` | Serializable request/response DTOs, model enums, errors, metrics |
| `NativeQuickDotAI.kt` | Kotlin wrapper around one native `CausalLmHandle` |
| `NativeCausalLm.kt` | Low-level JNI declarations |
| `LiteRTLm.kt` | LiteRT-LM engine wrapper for `ModelId.GEMMA4` |
| `NativeChatSession.kt` | Native chat-session helper |
| `LiteRTLmChatSession.kt` | LiteRT-LM chat-session helper |
| `ImageStore.kt` | Per-session image cache |
| `LlavaNextImageProcessor.kt` | Native multimodal preprocessing helper |
| `src/main/cpp/quickai_jni.cpp` | JNI bridge to `quick_dot_ai_api.h` |
| `src/main/cpp/CMakeLists.txt` | Builds `libquickai_jni.so` and links `libquick_dot_ai_api.so` |

## 🔌 Native Path

`NativeQuickDotAI` owns one native handle:

```text
NativeQuickDotAI
  └── NativeCausalLm.ensureLoaded()
      ├── System.loadLibrary("qnn_context")
      └── System.loadLibrary("quickai_jni")
            └── links/calls libquick_dot_ai_api.so
```

The native API surface is declared in `api/quick_dot_ai_api.h`.
The preferred calls are handle-based:

- `loadModelHandle`
- `runModelHandleWithMessagesStreaming`
- `runModelHandleWithJsonStreaming`
- `runMultimodalHandleStreaming`
- `cancelModelHandle`
- `destroyModelHandle`

## 🌗 LiteRT-LM Path

`LiteRTLm` is selected for `ModelId.GEMMA4` and takes a `.litertlm` file path
through `LoadModelRequest.modelPath`. `visionBackend != null` enables
multimodal calls for engines/models that support image input.

## 🧵 Threading Model

A `QuickDotAI` instance is not internally thread-safe. Host apps should drive a
loaded engine from one worker thread. `SampleTestAPP` follows this pattern with
a background dispatcher.

Streaming callbacks are delivered to the caller-provided `StreamSink`.
Apps that update UI must marshal callbacks to the main thread.

## 🧪 SampleTestAPP

`SampleTestAPP` is the current runnable Android sample. It links the
`:QuickDotAI` module directly; it does not start a REST service and does not
communicate over sockets.

## 🗺️ Planned Service Layer

The following pieces are design targets, not current Gradle modules:

| Planned component | Status |
|---|---|
| `LauncherApp` foreground-service bootstrap UI | Planned |
| `QuickAIService` remote foreground service | Planned |
| NanoHTTPD loopback REST server | Planned |
| `RequestDispatcher`, `ModelRegistry`, `ModelWorker` | Planned |
| Standalone REST client app | Planned |

When implemented, the service layer should wrap the same `QuickDotAI` AAR
contract rather than inventing a separate model API.

## 📦 Packaging

`apk-build-install.sh` performs the current full Android workflow:

1. Build native libraries with `./build.sh --platform=android --enable-qnn --clean`.
2. Install/copy native shared libraries through `apk_install_android.sh`.
3. Copy `.so` files into `Android/QuickDotAI/prebuilt_libs/`.
4. Run Gradle install for `:SampleTestAPP`.

Set `NDK_ROOT` inside `apk-build-install.sh` before using it on a new machine.

## 📎 Related Docs

- [QuickDotAI AAR API](QuickDotAI/README.md)
- [Native Streaming](AsyncAndStreaming.md)
- [Main README](../README.md)
