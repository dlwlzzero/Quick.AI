# How to Use XGrammar for Structured Generation

This document explains how to use XGrammar in Quick.AI for grammar-constrained text generation, ensuring 100% structural correctness of the output.

## Overview

[XGrammar](https://github.com/mlc-ai/xgrammar) is an open-source library for efficient, flexible, and portable structured generation. It leverages constrained decoding to ensure the generated output follows a specified grammar structure.

Quick.AI integrates XGrammar to enable:
- **JSON schema validation**: Generate outputs that conform to a specific JSON schema
- **Tool/function calling**: Structure outputs for tool invocation
- **Custom grammars**: Define custom context-free grammars for specific output formats

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  XGrammarManager (Singleton)                                    │
│  ├── TokenizerInfo (shared, created once per model)             │
│  ├── GrammarCompiler (shared, created once per model)            │
│  └── compiled_grammars_ (Map: tool_name → XGrammar)             │
│      ├── "alarm" → XGrammar                                      │
│      ├── "send_email" → XGrammar                                │
│      └── ...                                                     │
│                                                                  │
│  XGrammar                                                        │
│  ├── CompiledGrammar (compiled grammar)                         │
│  ├── GrammarMatcher (token masking)                             │
│  └── Bitmask (allowed token bitmask)                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `XGrammarManager` | Singleton manager that handles all grammar compilation and caching |
| `TokenizerInfo` | Contains vocabulary information for grammar compilation |
| `GrammarCompiler` | Compiles JSON schemas and grammars into executable form |
| `XGrammar` | Individual grammar instance with matcher and bitmask |
| `GrammarMatcher` | Applies token mask during generation |

## Usage Methods

There are two ways to use XGrammar in Quick.AI:

### 1. Pre-compile Method (Recommended)

Pre-compile all tool grammars at model load time using a `Toolset.json` file.

#### Flow

1. Place `Toolset.json` in your model directory
2. Call `loadModelHandle()` - automatically loads and compiles all tool grammars
3. A cache file (`Toolset.json.cache`) is created for faster subsequent loads
4. Call `runModelHandleWithTool()` with the tool name

#### Toolset.json Example

Create a `Toolset.json` file in your model directory:

```json
{
  "alarm": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["set", "delete", "snooze"]
      },
      "time": {
        "type": "string",
        "pattern": "^\\d{2}:\\d{2}$"
      },
      "message": {
        "type": "string"
      }
    },
    "required": ["action", "time"]
  },
  "send_email": {
    "type": "object",
    "properties": {
      "to": {
        "type": "string",
        "format": "email"
      },
      "subject": {
        "type": "string"
      },
      "body": {
        "type": "string"
      }
    },
    "required": ["to", "subject"]
  },
  "memo": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string"
      },
      "content": {
        "type": "string"
      }
    },
    "required": ["title"]
  }
}
```

#### Code Example

```c
#include "quick_dot_ai_api.h"

// Load model (automatically loads Toolset.json if present)
CausalLmHandle handle;
ErrorCode err = loadModelHandle(
    CAUSAL_LM_BACKEND_NPU,
    CAUSAL_LM_MODEL_GAUSS3_8_QNN,
    CAUSAL_LM_QUANTIZATION_W4A32,
    native_lib_dir,
    model_base_path,
    &handle
);

// Run inference with pre-compiled tool
const char *output;
err = runModelHandleWithTool(
    handle,
    "Set an alarm for 7am tomorrow",
    &output,
    "alarm",    // tool name from Toolset.json
    NULL        // schema not needed (pre-compiled)
);

printf("Output: %s\n", output);
// Output will be valid JSON conforming to "alarm" schema
// Example: {"action": "set", "time": "07:00", "message": "alarm for 7am tomorrow"}

// Cleanup
destroyModelHandle(handle);
```

### 2. Dynamic Compile Method

Register new tools at runtime with their JSON schemas.

#### Flow

1. Call `runModelHandleWithTool()` with a new tool name and schema
2. `XGrammarManager` checks if the tool exists
3. If not, compiles the schema and registers the tool
4. Runs inference with the newly compiled grammar

#### Code Example

```c
#include "quick_dot_ai_api.h"

// Load model
CausalLmHandle handle;
ErrorCode err = loadModelHandle(
    CAUSAL_LM_BACKEND_NPU,
    CAUSAL_LM_MODEL_GAUSS3_8_QNN,
    CAUSAL_LM_QUANTIZATION_W4A32,
    native_lib_dir,
    model_base_path,
    &handle
);

// Define a new tool schema at runtime
const char* search_schema = R"({
  "type": "object",
  "properties": {
    "query": {
      "type": "string"
    },
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100
    }
  },
  "required": ["query"]
})";

// Run inference with dynamic tool registration
const char *output;
err = runModelHandleWithTool(
    handle,
    "Search for recent news about AI",
    &output,
    "search",       // new tool name
    search_schema   // JSON schema for dynamic compilation
);

printf("Output: %s\n", output);
// Output: {"query": "recent news about AI", "limit": 10}

// The "search" tool is now registered and can be reused
err = runModelHandleWithTool(
    handle,
    "Find articles about machine learning",
    &output,
    "search",       // tool already registered
    NULL            // no schema needed
);

// Cleanup
destroyModelHandle(handle);
```

## API Reference

### runModelHandleWithTool

Run inference with grammar-constrained generation.

```c
ErrorCode runModelHandleWithTool(
    CausalLmHandle handle,       // Loaded model handle
    const char *inputTextPrompt, // Input prompt text
    const char **outputText,     // Output text buffer (owned by handle)
    const char *tool_name,       // Tool name (e.g., "alarm", "send_email")
    const char *tool_schema      // JSON schema (NULL if pre-compiled)
);
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `CausalLmHandle` | Handle returned by `loadModelHandle()` |
| `inputTextPrompt` | `const char*` | Input prompt text |
| `outputText` | `const char**` | Output buffer pointer (owned by handle, valid until next call) |
| `tool_name` | `const char*` | Name of the tool to use |
| `tool_schema` | `const char*` | JSON schema string (NULL if tool is pre-compiled) |

#### Return Value

| Error Code | Description |
|------------|-------------|
| `CAUSAL_LM_ERROR_NONE` | Success |
| `CAUSAL_LM_ERROR_INVALID_PARAMETER` | Invalid handle, NULL pointers, or tool not found without schema |
| `CAUSAL_LM_ERROR_NOT_INITIALIZED` | Model not initialized |
| `CAUSAL_LM_ERROR_UNKNOWN` | Failed to register tool or other error |

### XGrammarManager (C++ API)

For direct C++ usage, you can access the `XGrammarManager` singleton:

```cpp
#include "xgrammar_manager.h"
#include "xgrammar_wrapper.h"

// Initialize manager with tokenizer
causallm::XGrammarManager::Instance().initialize(tokenizer, vocab_size);

// Load toolset from file
causallm::XGrammarManager::Instance().loadToolset(
    "/path/to/Toolset.json",
    tokenizer,
    vocab_size
);

// Check if tool exists
bool has_tool = causallm::XGrammarManager::Instance().hasTool("alarm");

// Get grammar for a tool
causallm::XGrammar* grammar = causallm::XGrammarManager::Instance().getGrammar("alarm");

// Register a new tool dynamically
causallm::XGrammarManager::Instance().registerTool("new_tool", json_schema);

// Reset grammar state for next generation
causallm::XGrammarManager::Instance().resetGrammar("alarm");

// Get all registered tool names
std::vector<std::string> tools = causallm::XGrammarManager::Instance().getToolNames();

// Clear all grammars
causallm::XGrammarManager::Instance().clear();
```

## Cache Mechanism

XGrammar uses a caching mechanism to speed up subsequent loads:

1. When `loadToolset()` is called, it checks for `Toolset.json.cache`
2. If cache exists and is valid, grammars are loaded from cache (fast)
3. If cache doesn't exist or is invalid, grammars are compiled and cache is saved

```
Model Directory/
├── Toolset.json        # Tool definitions
├── Toolset.json.cache  # Compiled grammar cache (auto-generated)
├── config.json
├── nntr_config.json
└── model.bin
```

## Performance Considerations

| Method | Initial Load | Subsequent Calls | Use Case |
|--------|--------------|------------------|----------|
| Pre-compile | Slower (compile all) | Fast (from cache) | Known tools, production |
| Dynamic | Fast (no pre-compile) | Slower (compile on first use) | Runtime tool discovery |

### Recommendations

1. **Use Pre-compile for known tools**: Place all your tool definitions in `Toolset.json` for optimal performance
2. **Use Dynamic for ad-hoc tools**: When tools are discovered at runtime or vary per user
3. **Cache is your friend**: The first load compiles and caches; subsequent loads are much faster

## JSON Schema Support

XGrammar supports standard JSON Schema features:

- `type`: `string`, `integer`, `number`, `boolean`, `object`, `array`
- `properties`: Object property definitions
- `required`: Required property names
- `enum`: Enumerated values
- `minimum` / `maximum`: Numeric constraints
- `minLength` / `maxLength`: String length constraints
- `pattern`: Regex pattern for strings
- `format`: `email`, `uri`, `date`, `date-time`

## Troubleshooting

### Tool not found error

```
Error: Tool 'my_tool' not found and no schema provided
```

**Solution**: Either add the tool to `Toolset.json` or provide a schema when calling `runModelHandleWithTool()`.

### Grammar compilation failed

```
Error: Failed to register tool 'my_tool'
```

**Solution**: Check that your JSON schema is valid. Use a JSON schema validator to verify.

### Output doesn't match schema

**Solution**: Ensure the model is capable of generating the expected output format. Some models may need fine-tuning for structured output.

## Related Documentation

- [XGrammar Official Documentation](https://xgrammar.mlc.ai/docs/)
- [XGrammar GitHub Repository](https://github.com/mlc-ai/xgrammar)
- [JSON Schema Specification](https://json-schema.org/)
