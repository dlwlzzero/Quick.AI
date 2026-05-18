# XGrammar Reference 🧩

This document is the XGrammar reference for Quick.AI. It explains the runtime
components, toolset files, cache behavior, and native API contract. For
end-to-end Chat tab, OpenAI JSON, and XGrammar usage examples, start with
[Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md).

## 🔎 Overview

[XGrammar](https://github.com/mlc-ai/xgrammar) provides constrained decoding:
Quick.AI masks invalid next tokens during sampling so generated text follows a
target grammar or JSON schema.

Quick.AI currently uses XGrammar for:

- JSON-schema-constrained output
- tool/function-call shaped output
- reusable grammars loaded from model-local toolset files

This is separate from OpenAI JSON `tools`. OpenAI JSON `tools` are rendered
through the model's chat template; XGrammar enforces the output shape during
token selection.

## 🧱 Architecture

```text
XGrammarManager
  - TokenizerInfo, shared per loaded model
  - GrammarCompiler, shared per loaded model
  - compiled_grammars_: tool_name -> XGrammar

XGrammar
  - CompiledGrammar
  - GrammarMatcher
  - bitmask of allowed next tokens

Model sampler
  - asks XGrammar for the next-token mask
  - samples from the masked logits
  - accepts the sampled token into the GrammarMatcher
```

| Component | Responsibility |
|---|---|
| `XGrammarManager` | Owns shared tokenizer/compiler state and cached tool grammars. |
| `TokenizerInfo` | Stores the model vocabulary in the format XGrammar needs. |
| `GrammarCompiler` | Compiles JSON schemas, EBNF grammars, or regex patterns. |
| `XGrammar` | Holds one compiled grammar and its matcher state. |
| `GrammarMatcher` | Tracks grammar progress and produces allowed-token masks. |

## 🔄 Runtime Flow

At model load time, Quick.AI initializes the XGrammar manager from the loaded
tokenizer. If the model directory contains `Toolset.json`, Quick.AI precompiles
the listed schemas and stores them by tool name.

At inference time, `runModelHandleWithTool()`:

1. Looks up the requested tool name in `XGrammarManager`.
2. Registers `tool_schema` dynamically if the tool is missing and a schema was
   provided.
3. Attaches the resulting `XGrammar` instance to the model.
4. Runs inference while the sampler applies grammar masks.
5. Resets the model grammar state after generation.

## 📦 Toolset Files

Place `Toolset.json` next to the model files used by `loadModelHandle()`.

```json
{
  "tool_name": {
    "type": "object",
    "properties": {
      "field": { "type": "string" }
    },
    "required": ["field"]
  }
}
```

The top-level object maps a tool name to a JSON Schema. Tool names are later
passed to `runModelHandleWithTool()`.

## 🧠 Native API

```c
ErrorCode runModelHandleWithTool(CausalLmHandle handle,
                                 const char *inputTextPrompt,
                                 const char **outputText,
                                 const char *tool_name,
                                 const char *tool_schema);
```

| Parameter | Meaning |
|---|---|
| `handle` | Loaded model handle. |
| `inputTextPrompt` | Prompt to run with grammar constraints. |
| `outputText` | Receives the generated output pointer. |
| `tool_name` | Name of a precompiled or dynamically registered schema. |
| `tool_schema` | JSON Schema string used when `tool_name` is not already registered. |

Return values follow the common `ErrorCode` contract in the
[C API Reference](../api/README.md).

## 🧰 C++ Manager API

Direct C++ integrations can use `causallm::XGrammarManager::Instance()` from
`src/xgrammar/xgrammar_manager.h`.

| Method | Purpose |
|---|---|
| `initialize(tokenizer, vocab_size)` | Build shared tokenizer/compiler state. |
| `loadToolset(path, tokenizer, vocab_size)` | Load and compile `Toolset.json`. |
| `hasTool(name)` | Check whether a grammar is registered. |
| `getGrammar(name)` | Fetch a compiled `XGrammar`. |
| `registerTool(name, schema)` | Compile and register a schema at runtime. |
| `resetGrammar(name)` | Reset matcher state after a run. |
| `getToolNames()` | List registered tool names. |
| `clear()` | Drop compiled grammars and shared state. |

## 🗂️ Cache Behavior

XGrammar compilation can be expensive. Quick.AI uses a sidecar cache:

```text
model/
  Toolset.json
  Toolset.json.cache
```

Load behavior:

1. If `Toolset.json.cache` exists and is valid, grammars are loaded from cache.
2. If the cache is missing or incomplete, Quick.AI compiles from `Toolset.json`.
3. After successful compilation, Quick.AI saves a fresh cache file.

Delete `Toolset.json.cache` when changing schemas during development.

## ✅ JSON Schema Support

XGrammar supports the schema features used by Quick.AI tool definitions:

- object, string, number, integer, boolean, array
- `properties`
- `required`
- `enum`
- `pattern`
- nested objects and arrays

Keep schemas narrow and explicit for best constrained-decoding behavior.

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Tool not found | `tool_name` is absent and `tool_schema` is null. | Add the tool to `Toolset.json` or pass a schema on first use. |
| Compilation failed | Unsupported or malformed JSON Schema. | Validate the schema and start with a smaller object shape. |
| Output is valid JSON but semantically wrong | Grammar only constrains structure. | Add stronger prompt instructions or narrower schema fields. |
| First load is slow | Toolset schemas are being compiled. | Keep the generated `Toolset.json.cache` for later loads. |
| Cache seems stale | The schema changed but cache was reused. | Delete `Toolset.json.cache` and reload the model. |

## 📎 Related Docs

- [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md)
- [Chat Templates](ChatTemplate.md)
- [C API Reference](../api/README.md)
- [XGrammar Official Documentation](https://xgrammar.mlc.ai/docs/)
- [XGrammar GitHub Repository](https://github.com/mlc-ai/xgrammar)
