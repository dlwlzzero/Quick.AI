from gemma4_qc.utils import RopeEmbedding
import torch
import functools

def get_dummy_data(mode, batchsize, seqlen, model_cfg, device, dtype=None):
    """
    Generate dummy inputs for profiling or compilation, matching the
    package Gemma4TextModel.forward signature.

    Args:
        mode: "prefill" (input_seqlen=256) or "generation" (input_seqlen=1).
        batchsize: Batch size.
        seqlen: Total sequence length (input + cache).
        model_cfg: text_config with model dimensions.
        device: Target device.
        dtype: Tensor dtype (defaults to float32).

    Returns:
        dict with keys: inputs_embeds, per_layer_inputs, attention_mask,
        sliding_attention_mask, position_ids, past_key_values

    Notes:
        - Full-attention KV cache length: cache_length tokens.
        - Sliding-attention KV cache length: sliding_window tokens when
          seqlen > sliding_window (cache is bounded by the window). After
          attention concatenates current-step KV with past KV, total KV =
          sliding_window + input_seqlen — which is what the sliding mask
          is sized to cover.
        - KV-shared layers (idx >= num_hidden_layers - num_kv_shared_layers)
          read KV from their source layer via shared_kv_store; they do NOT
          appear in past_key_values at all (no None placeholders).
        - In prefill mode (return_kv_cache_only graph), shared layers are
          skipped entirely, so their per_layer_inputs are set to None.
    """
    input_seqlen = {"prefill": 256, "generation": 1}[mode]
    context_seqlen = {"prefill": seqlen, "generation": seqlen + 512}[mode]
    cache_length = context_seqlen - input_seqlen
    dtype = dtype or torch.float32

    sliding_window = model_cfg.sliding_window
    sliding_cache_length = sliding_window
    # if seqlen <= sliding_window:
    #     sliding_cache_length = cache_length
    # else:
    #     actual_garbage_len = {"prefill": 0, "generation": 1}[mode]
    #     sliding_cache_length = sliding_window - actual_garbage_len

    num_kv_shared_layers = getattr(model_cfg, "num_kv_shared_layers", 0)
    first_kv_shared_layer_idx = model_cfg.num_hidden_layers - num_kv_shared_layers
    if first_kv_shared_layer_idx < 0:
        first_kv_shared_layer_idx = model_cfg.num_hidden_layers + 1

    dummy_inputs_embeds = torch.randn(
        batchsize, input_seqlen, model_cfg.hidden_size,
        dtype=dtype, device=device,
    )

    dummy_per_layer_inputs = None
    if model_cfg.hidden_size_per_layer_input:
        dummy_per_layer_inputs = []
        for layer_idx in range(model_cfg.num_hidden_layers):
            is_kv_shared = (
                num_kv_shared_layers > 0
                and layer_idx >= first_kv_shared_layer_idx
            )
            if not is_kv_shared or mode == "generation":
                dummy_per_layer_inputs.append(
                    torch.randn(
                        batchsize, input_seqlen, model_cfg.hidden_size_per_layer_input,
                        dtype=dtype, device=device,
                    )
                )

    # 4D masks sized for (input_seqlen, past_kv_len + input_seqlen)
    dummy_attention_mask = torch.zeros(
        batchsize, 1, input_seqlen, cache_length + input_seqlen,
        dtype=dtype, device=device,
    )
    dummy_sliding_attention_mask = torch.zeros(
        batchsize, 1, input_seqlen, sliding_cache_length + input_seqlen,
        dtype=dtype, device=device,
    )

    dummy_position_ids = torch.arange(
        cache_length, cache_length + input_seqlen, device=device,
    ).unsqueeze(0).expand(batchsize, -1)
    rope_fn = make_rope_fn(model_cfg, device=str(device), max_length=seqlen + 1024)
    dummy_rope_4tuple = rope_fn(dummy_position_ids)

    # past_key_values contains entries ONLY for non-shared layers, in order.
    global_head_dim = model_cfg.global_head_dim or model_cfg.head_dim
    dummy_past_key_values = []
    for layer_idx in range(model_cfg.num_hidden_layers):
        is_kv_shared = (
            num_kv_shared_layers > 0
            and layer_idx >= first_kv_shared_layer_idx
        )
        if is_kv_shared:
            continue

        layer_type = model_cfg.layer_types[layer_idx]
        is_sliding = layer_type == "sliding_attention"
        head_dim = model_cfg.head_dim if is_sliding else global_head_dim
        kv_cache_len = sliding_cache_length if is_sliding else cache_length

        keys = tuple(
            torch.zeros(batchsize, 1, head_dim, kv_cache_len, dtype=dtype, device=device)
            for _ in range(model_cfg.num_key_value_heads)
        )
        values = tuple(
            torch.zeros(batchsize, 1, kv_cache_len, head_dim, dtype=dtype, device=device)
            for _ in range(model_cfg.num_key_value_heads)
        )
        dummy_past_key_values.append((keys, values))

    return {
        "input_embeds": dummy_inputs_embeds,
        "per_layer_inputs": dummy_per_layer_inputs,
        "attention_mask": dummy_attention_mask,
        "sliding_attention_mask": dummy_sliding_attention_mask,
        "position_ids": dummy_rope_4tuple,
        "past_key_values": dummy_past_key_values,
    }

def rope_fn(position_ids, global_rope, sliding_rope):
    gc, gs = global_rope.get_embedding(position_ids)
    sc, ss = sliding_rope.get_embedding(position_ids)

    return (gc, gs, sc, ss)

def make_rope_fn(text_config, device="cpu", max_length=4096):
    """
    Create a rope_fn callable for Gemma4TextForCausalLM.generate().

    Returns a function: position_ids -> (global_cos, global_sin, sliding_cos, sliding_sin)
    Each element has shape (batch, 1, seq, head_dim//2).
    """
    global_head_dim = (
        text_config.global_head_dim
        if text_config.global_head_dim is not None
        else text_config.head_dim
    )
    global_theta = text_config.rope_parameters.get("full_attention", {}).get("rope_theta", 1000000.0)
    sliding_theta = text_config.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0)

    global_rope = RopeEmbedding(device=device, head_dim=global_head_dim, max_length=max_length, theta=global_theta)
    sliding_rope = RopeEmbedding(device=device, head_dim=text_config.head_dim, max_length=max_length, theta=sliding_theta)

    return functools.partial(rope_fn, global_rope=global_rope, sliding_rope=sliding_rope)


def preprocess_input(text, processor, device="cpu"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]

    # Process input
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(device)
    inputs.pop("mm_token_type_ids")
    return inputs

def postprocess_outputs(inputs, outputs, processor):
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    output = processor.parse_response(response)
    return output["content"], output

def get_input_output_names(export_mode, model_cfg, **kwargs):
    """
    Return input/output names for ONNX/QNN export that match the non-None
    inputs/outputs produced by get_dummy_data and consumed by the package
    Gemma4TextModel.forward.

    Conventions:
        - KV-shared layers (idx >= num_hidden_layers - num_kv_shared_layers)
          do NOT appear in past_key/past_value I/O (they reuse source-layer
          KV via shared_kv_store).
        - Per-layer inputs exist only when hidden_size_per_layer_input > 0.
          In prefill shared layers are skipped
          so their per_layer_inputs are omitted.
        - RoPE is a 4-tuple: (global_cos, global_sin, sliding_cos, sliding_sin).
        - Prefill has no past_kv inputs; generation does.
        - Output KV per non-shared layer is always emitted (new-kv-only for
          compilation via return_new_key_value_only=True).

    Args:
        export_mode: "prefill" or "generation".
        model_cfg: text_config with model dimensions.
    """
    num_layers = model_cfg.num_hidden_layers
    num_kv_heads = model_cfg.num_key_value_heads
    num_kv_shared = getattr(model_cfg, "num_kv_shared_layers", 0)
    first_shared_idx = num_layers - num_kv_shared
    if first_shared_idx < 0:
        first_shared_idx = model_cfg.num_hidden_layers + 1
    non_shared_indices = [i for i in range(num_layers) if i < first_shared_idx]

    def _pkv_names(sfx, indices):
        names = []
        for i in indices:
            names.extend([f"past_key_{i}_h{h}_{sfx}" for h in range(num_kv_heads)])
            names.extend([f"past_value_{i}_h{h}_{sfx}" for h in range(num_kv_heads)])
        return names

    input_names = ["input_embeds", "attention_mask", "sliding_attention_mask"]
    input_names += [
        "position_ids_cos",
        "position_ids_sin",
        "swa_position_ids_cos",
        "swa_position_ids_sin",
    ]
    if getattr(model_cfg, "hidden_size_per_layer_input", 0):
        if export_mode == "generation":
            per_layer_indices = range(num_layers)
        else:
            per_layer_indices = non_shared_indices
        input_names += [f"per_layer_inputs_{i}" for i in per_layer_indices]
    input_names += _pkv_names("in", non_shared_indices)

    output_names = []
    if export_mode == "generation":
        output_names.append("logits")
    output_names += _pkv_names("out", non_shared_indices)
    return input_names, output_names
