# Quick.AI Architecture

> This document was moved from the root `README.md` to keep the README concise for new users. See [README.md](../README.md) for the project overview.

## Plugin System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  nntrainer (submodule)                                       │
│  ├── main.cpp + Factory singleton                            │
│  │     ├── Qwen3ForCausalLM      (built-in)                  │
│  │     ├── GptOssForCausalLM     (built-in)                  │
│  │     └── ...                                               │
│  │                                                           │
│  quick-dot-ai (this repo)                                    │
│  ├── quick_dot_ai            (standalone executable)         │
│  │     └── Gauss2_5ForCausalLM ◄── statically linked (always │
│  │                                 available, no LD_PRELOAD) │
│  └── libquick_dot_ai.so          (plugin for LD_PRELOAD)     │
│        __attribute__((constructor)) runs before main()       │
│        → Factory::Instance().registerModel(...)              │
└──────────────────────────────────────────────────────────────┘
```

Custom models use `__attribute__((constructor))` to register with the `Factory` singleton **before `main()` starts**. The standalone executable (`quick_dot_ai`) statically links the custom models via `link_whole`, so they are always available without `LD_PRELOAD`. A shared library is also built for plugin mode with the original `nntr_causallm`.

This means:

- nntrainer's `main.cpp` is used as-is — never copied or modified
- When nntrainer updates, nothing in this repo breaks
- Multiple custom models can be added independently

## Self-Registration Mechanism

```cpp
__attribute__((constructor)) static void register_my_models() {
  causallm::Factory::Instance().registerModel(
    "MyModelForCausalLM",
    [](causallm::json cfg, causallm::json generation_cfg,
       causallm::json nntr_cfg) {
      return std::make_unique<causallm::MyModel>(
        cfg, generation_cfg, nntr_cfg);
    });
}
```

The `__attribute__((constructor))` ensures the registration function runs before `main()`, so the model is available immediately at program start.

## Build Artifacts

| Target | Output | Purpose |
|--------|--------|---------|
| `src/quick_dot_ai` | Standalone executable | Built-in custom models, no plugin needed |
| `src/libquick_dot_ai.so` | Shared library | `LD_PRELOAD` plugin for original `nntr_causallm` |

## Related Documentation

- [How to Create a Custom Model](../README.md#how-to-create-a-custom-model) — Step-by-step plugin authoring guide
- [Android Architecture](../Android/Architecture.md) — Android service and AAR architecture
