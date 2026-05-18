# gemma4_qc 🧪

Quantization-ready reimplementation of the Gemma4 text decoder. All arithmetic ops are wrapped as `nn.Module` for quantizer insertion, with optional Conv2d-based (SHA) projections for NPU compilation.

## Install

From the Quick.AI repository root:

```bash
pip install -e gemma_python/
```

Or from inside this directory:

```bash
pip install -e .
```

## Quick Start

```python
import torch
from transformers import AutoConfig, AutoProcessor
from gemma4_qc.gemma4_text import Gemma4TextForCausalLM, make_rope_fn

MODEL_PATH = "path/to/gemma-4-E2B-it-textonly-untied"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config + model
config = AutoConfig.from_pretrained(MODEL_PATH)
config._attn_implementation = "eager"

model = Gemma4TextForCausalLM.from_pretrained(MODEL_PATH, config=config)
model.eval().to(device)
model.set_rope_fn(make_rope_fn(config, device=device))

# Prepare input using processor
processor = AutoProcessor.from_pretrained(MODEL_PATH)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of South Korea?"},
]
text = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=text, return_tensors="pt").to(device)
inputs.pop("mm_token_type_ids")

# Generate
input_ids = inputs["input_ids"]
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=256, do_sample=False)

response = processor.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(processor.parse_response(response)["content"])
```

## Manual Forward

```python
# prepare_inputs returns: (inputs_embeds, per_layer_inputs, attention_mask_4d, sliding_mask_4d)
embeds, ple, attn_mask, sliding_mask = model.prepare_inputs(input_ids)
position_ids = rope_fn(torch.arange(input_ids.shape[1]).unsqueeze(0))

output = model(
    inputs_embeds=embeds,
    per_layer_inputs=ple,
    position_ids=position_ids,
    attention_mask=attn_mask,
    sliding_attention_mask=sliding_mask,
)
```

## Conv2d / SHA Mode

Convert all layers to per-head Conv2d projections for NPU compilation:

```python
model.enable_conv()
output = model.generate(input_ids, max_new_tokens=50)
```

## Compilation-Style Forward

Prefill graph:

```python
output = model(
    inputs_embeds=embeds,
    position_ids=position_ids,
    per_layer_inputs=ple,
    attention_mask=external_4d_mask,        # (bsz, 1, q_len, kv_len)
    sliding_attention_mask=external_sw_mask, # (bsz, 1, q_len, kv_len)
    past_key_values=past_kv,                # list of (keys_tuple, vals_tuple)
    use_cache=True,
    return_new_key_value_only=True,
)
```

Generation graph:

```python
output = model(
    inputs_embeds=embeds,
    position_ids=position_ids,
    per_layer_inputs=ple,
    attention_mask=external_4d_mask,        # (bsz, 1, q_len, kv_len)
    sliding_attention_mask=external_sw_mask, # (bsz, 1, q_len, kv_len)
    past_key_values=past_kv,                # list of (keys_tuple, vals_tuple)
    use_cache=True,
)
```

## KV Cache Format

```python
past_key_values = [           # one per layer (None for KV-shared layers)
    (
        (k_head0, k_head1, ...), # keys:   (bsz, 1, head_dim, seq_len)
        (v_head0, v_head1, ...), # values: (bsz, 1, seq_len, head_dim)
    ),
    ...
]
```
