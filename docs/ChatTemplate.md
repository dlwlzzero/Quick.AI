# Chat Templates 💬

Quick.AI uses chat templates to convert OpenAI-style chat requests into the
model-specific prompt strings expected by nntrainer/LiteRT-LM style models.
The behavior is intentionally close to Hugging Face
`tokenizer.apply_chat_template()`.

## 🔎 Template Discovery

For a model directory, Quick.AI looks for a template in this order:

1. `<model_path>/chat_template.jinja`
2. `<model_path>/tokenizer_config.json`, field `chat_template`
3. Built-in fallback formatting for selected architectures

`tokenizer_config.json.chat_template` may be a string, an object of named
templates, or an array converted into named templates. When named templates are
available, `tool_use` is selected for requests containing tools; otherwise
`default` is preferred.

Special tokens are loaded from `tokenizer_config.json` and
`special_tokens_map.json` when present.

## 🧱 Native API Integration

Chat templates are used by these C API paths:

| API | Input |
|---|---|
| `applyChatTemplate()` | `CausalLMChatMessage[]` |
| `runModelHandleStreaming()` | Raw prompt string (uses chat template when `g_use_chat_template` is true and input is not already formatted) |
| `runModelHandleWithMessages()` | `CausalLMChatMessage[]` |
| `runModelHandleWithMessagesStreaming()` | `CausalLMChatMessage[]` |
| `runModelHandleWithJsonStreaming()` | OpenAI-style JSON string |

For JSON streaming, the loaded model must provide a usable chat template. If no
template is available, `runModelHandleWithJsonStreaming()` returns
`CAUSAL_LM_ERROR_UNSUPPORTED`.

## 📦 OpenAI JSON Shape

```json
{
  "messages": [
    { "role": "developer", "content": "You can call tools." },
    { "role": "user", "content": "Call mom." }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "call",
        "description": "Make a phone call.",
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
```

Legacy OpenAI `functions` is accepted as an alias for raw function schemas.

## 🧑‍💻 Usage Examples

End-to-end Chat tab, OpenAI tab, native messages, and JSON streaming examples
live in [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md).

## ⚠️ Notes

- `developer` role handling depends on the template and renderer options.
- `tool` and function-call message formatting depends on the model template.
- Native message APIs use text-only `role/content` pairs; Android
  `QuickAiChatMessage` can also carry image parts for multimodal methods.
- Template files should live next to the model config files used by
  `loadModelHandle()`.

## 📎 Related Docs

- [Chat and OpenAI Usage Examples](ChatAndOpenAIUsage.md)
- [C API Reference](../api/README.md)
- [QuickDotAI AAR API](../Android/QuickDotAI/README.md)
