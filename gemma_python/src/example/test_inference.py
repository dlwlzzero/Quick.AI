"""
Example: Using the gemma4_qc package for inference.

Usage:
    pip install -e /path/to/gemma4-modeling
    python test_inference.py
"""
import copy
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from gemma4_qc.gemma4_text import (
    Gemma4TextForCausalLM,
    make_rope_fn,
    get_dummy_data,
    preprocess_input,
    postprocess_outputs
)


# MODEL_PATH = "/group-volume/models/gemma-4-E2B-it-textonly-untied"
# IS_FOLDED = False

MODEL_PATH = "/group-volume/models/gemma-4-E2B-it-textonly-untied-foldedple"
IS_FOLDED = True

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"

    # --- 1. Load config ---
    text_config = AutoConfig.from_pretrained(MODEL_PATH)
    text_config._attn_implementation = "eager"
    text_config.enable_moe_block = False
    text_config.is_folded = IS_FOLDED
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # --- 2. Create model,set rope function and prepare inputs ---
    model = Gemma4TextForCausalLM.from_pretrained(MODEL_PATH, config=text_config)
    model.eval()
    model.to(device)

    rope_fn = make_rope_fn(text_config, device=device)
    model.set_rope_fn(rope_fn)

    # --- 3. Single forward pass (prepare_inputs + forward) ---
    print("=== Single forward pass (prepare_inputs + forward) ===")

    # input_text = "What is the capital of South Korea?"
    # input_text = "무지개를 주제로 한 짧은 글을 써 줘."
    # input_text = "What is the rainbow?"
    input_text = "What is 1/3 + 1/4?"

    inputs = preprocess_input(input_text, processor, device=device)
    input_ids = inputs["input_ids"]
    input_ids = input_ids.to(device)
    inputs_embeds, per_layer_inputs, attn_mask, sliding_mask = model.prepare_inputs(input_ids, is_folded=IS_FOLDED)
    position_ids = torch.arange(len(input_ids[0])).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            position_ids=rope_fn(position_ids),
            per_layer_inputs=per_layer_inputs,
            attention_mask=attn_mask,
            sliding_attention_mask=sliding_mask,
        )
    print(f"  logits shape: {output.logits.shape}")
    print(f"  predicted token: {output.logits[0, -1].argmax().item()}")

    # --- 4. HF generate() (uses prepare_inputs_for_generation + QC KV cache) ---
    print("\n=== HF generate() ===")
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    print(f"  input:     {input_ids[0].tolist()}")
    print(f"  generated: {generated[0].tolist()}")
    print(f"  new tokens: {generated[0, input_ids.shape[1]:].tolist()}")
    print(f"  generated text : {postprocess_outputs(inputs, generated, processor)}")

    # --- 5. HF generate() with Conv2d path ---
    print("\n=== HF generate() with Conv2d ===")
    model.enable_conv()
    with torch.no_grad():
        generated_conv = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    print(f"  new tokens (conv): {generated_conv[0, input_ids.shape[1]:].tolist()}")
    print(f"  generated text (conv) : {postprocess_outputs(inputs, generated_conv, processor)}")
    print(f"  matches linear: {(generated[0] == generated_conv[0]).all().item()}")

    # --- 6. Compilation-style forward (input + cache, new KV only) ---
    print("\n=== Compilation-style forward (generation) ===")

    generation_inputs = get_dummy_data("generation", 1, 1024, model.config, device, model.dtype)
    model.return_kv_cache_only = False

    # generation graph
    with torch.no_grad():
        output = model(
            inputs_embeds=generation_inputs["input_embeds"],
            position_ids=generation_inputs["position_ids"],
            per_layer_inputs=generation_inputs["per_layer_inputs"],
            attention_mask=generation_inputs["attention_mask"],
            sliding_attention_mask=generation_inputs["sliding_attention_mask"],
            past_key_values=generation_inputs["past_key_values"],
            use_cache=True,
            return_new_key_value_only=True,
        )
    input_len = generation_inputs["input_embeds"].shape[-2]
    print(f"  logits shape: {output.logits.shape}")
    print(f"  KV layers: {len(output.past_key_values)}")
    print(f"  new KV seq len: {output.past_key_values[0][0][0].shape[-1]} (should be {input_len})")

    print("\n=== Compilation-style forward (prefill) ===")

    prefill_inputs = get_dummy_data("prefill", 1, 1024, model.config, device, model.dtype)
    model.return_kv_cache_only = True

    # prefill graph
    with torch.no_grad():
        output_kv = model(
            inputs_embeds=prefill_inputs["input_embeds"],
            position_ids=prefill_inputs["position_ids"],
            per_layer_inputs=prefill_inputs["per_layer_inputs"],
            attention_mask=prefill_inputs["attention_mask"],
            sliding_attention_mask=prefill_inputs["sliding_attention_mask"],
            past_key_values=prefill_inputs["past_key_values"],
            use_cache=True,
            return_new_key_value_only=True,
        )
    input_len = prefill_inputs["input_embeds"].shape[-2]
    print(f"  logits: {output_kv.logits}")
    print(f"  KV layers: {len(output_kv.past_key_values)}")
    print(f"  new KV seq len: {output_kv.past_key_values[0][0][0].shape[-1]} (should be {input_len})")

    print("\nDone.")


if __name__ == "__main__":
    main()
