git# ChatTemplate

`ChatTemplate` adapts OpenAI-style chat inputs into the model-specific prompt
strings expected by Quick.AI runners. It is intended to behave like Hugging Face
`tokenizer.apply_chat_template()` while keeping the public C API unchanged.

The current integration is limited to the native runner path in `main.cpp`.
When `nntr_config.json` contains `chat_input`, the runner applies a chat
template before tokenization and generation.

## Template discovery

For a model directory passed to `quick_dot_ai_run`, Quick.AI looks for a chat
template in this order:

1. `<model_path>/chat_template.jinja`
2. `<model_path>/tokenizer_config.json`, field `chat_template`
3. Built-in Function Gemma template, only for `Gemma3ForCausalLM` with
   `chat_input`

`tokenizer_config.json.chat_template` can be either a string template or an
object of named templates. When named templates are used, `tool_use` is selected
automatically if tools are present; otherwise `default` is preferred.

Special tokens are loaded from `tokenizer_config.json` and
`special_tokens_map.json` when present.

## Runner usage

Add `chat_input` to `nntr_config.json`:

```json
{
  "chat_input": {
    "messages": [
      { "role": "system", "content": "You are concise." },
      { "role": "user", "content": "Hello" },
      { "role": "assistant", "content": "Hi." },
      { "role": "user", "content": "Repeat that." }
    ]
  }
}
```

Run the model as usual:

```bash
./build/quick_dot_ai_run ./res/qwen3/qwen3-4b/
```

If a prompt is passed as the second CLI argument, it is used as a raw prompt and
`chat_input` is not applied:

```bash
./build/quick_dot_ai_run ./res/qwen3/qwen3-4b/ "raw prompt"
```

## Request format

`ChatTemplate::apply()` accepts either:

- an array of OpenAI-style messages
- an object containing a `messages` array

Supported message roles:

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

Message `content` can be a string. Text-only content parts are also accepted:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Hello" },
    { "type": "text", "text": " again" }
  ]
}
```

Non-text content parts are rejected for now.

## Generation prompt behavior

By default, `ChatTemplate` appends the assistant generation prompt unless the
last message is already an assistant message.

The request can override this behavior with:

```json
{
  "add_generation_prompt": false,
  "messages": [
    { "role": "user", "content": "Hello" }
  ]
}
```

`continue_final_message` is recognized but not supported by the runner yet. It
currently returns an error instead of silently producing a possibly incorrect
prompt.

## Tool and function calling

Tools are normalized before they are passed to the template. The preferred input
is the current OpenAI `tools` format:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city name"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "messages": [
    { "role": "developer", "content": "You can call functions." },
    { "role": "user", "content": "Weather in Seoul?" }
  ]
}
```

Raw function schemas are also accepted and converted into OpenAI tool objects:

```json
{
  "tools": [
    {
      "name": "get_current_temperature",
      "description": "Gets the current temperature for a given location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [
    { "role": "developer", "content": "You can call functions." },
    { "role": "user", "content": "Weather in Seoul?" }
  ]
}
```

Legacy OpenAI `functions` is accepted as an alias for raw function schemas:

```json
{
  "functions": [
    {
      "name": "get_current_temperature",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        }
      }
    }
  ],
  "messages": [
    { "role": "developer", "content": "You can call functions." },
    { "role": "user", "content": "Weather in Seoul?" }
  ]
}
```

## Multi-turn tool calls

Assistant tool calls and tool responses can be represented with OpenAI-style
`tool_calls` and `tool_call_id`.

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "messages": [
    { "role": "developer", "content": "You can call functions." },
    { "role": "user", "content": "Weather in Seoul?" },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_current_temperature",
            "arguments": { "location": "Seoul" }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "12 C"
    },
    { "role": "user", "content": "Summarize it." }
  ]
}
```

If a `tool` message does not include `name`, `ChatTemplate` resolves it from the
preceding assistant `tool_calls` entry with the same `tool_call_id`.

Legacy assistant `function_call` is converted into a single `tool_calls` entry.

## C++ usage

Use `ChatTemplate::Load()` for model-provided templates:

```cpp
#include "chat_template.h"

quick_dot_ai::ChatTemplate chat_template =
  quick_dot_ai::ChatTemplate::Load(model_path);

std::string prompt = chat_template.apply(nntr_cfg["chat_input"]);
```

Use the built-in Function Gemma template when no model template exists:

```cpp
quick_dot_ai::ChatTemplate chat_template =
  quick_dot_ai::ChatTemplate::LoadBuiltin(
    quick_dot_ai::ChatTemplate::Builtin::FunctionGemma);

std::string prompt = chat_template.apply(chat_input);
```

Optional rendering controls are available through `ChatTemplate::Options`:

```cpp
quick_dot_ai::ChatTemplate::Options options;
options.generation_prompt =
  quick_dot_ai::ChatTemplate::Options::GenerationPromptMode::Never;
options.developer_role_policy =
  quick_dot_ai::ChatTemplate::Options::DeveloperRolePolicy::MergeIntoSystem;

std::string prompt = chat_template.apply(chat_input, options);
```

## Developer role handling

File-based Hugging Face templates default to merging `developer` messages into
`system` messages, because many templates do not know the `developer` role.

The built-in Function Gemma template preserves `developer`, matching the
existing Function Gemma prompt format.

This can be overridden with `ChatTemplate::Options::developer_role_policy`.

## Current limitations

- Integration is currently in `main.cpp`; the C API is unchanged.
- Only text chat content is supported.
- `continue_final_message` is not supported yet.
- Images, audio, and other multimodal content parts are rejected.
- Tool execution is out of scope; `ChatTemplate` only formats prompts.

## Implementation files

- `chat_template.h`: public C++ interface
- `chat_template.cpp`: template loading, request normalization, rendering
- `main.cpp`: native runner integration
- `third_party/minja/`: vendored Jinja-compatible chat template renderer
