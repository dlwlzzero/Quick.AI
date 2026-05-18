import math
import functools
from typing import List, Optional, Tuple, Union

# import gemma4_qc.elementwise_ops as op
import aimet_torch.nn.modules.custom as op
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import logging
from transformers import Gemma4Config
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.nn.true_quant import _DispatchMixin, _quantize_dequantize_if_applicable, _quantize_if_applicable

logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0, mask_neg: float = -100.0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    #mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask = torch.full((tgt_len, tgt_len), torch.tensor(mask_neg, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, mask_neg: float = -100.0, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    #return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_neg)


class Gemma4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, with_scale = True, dtype=None, device=None):
        super().__init__()
        self.variance_epsilon = eps
        self.with_scale = with_scale

        self.cast = op.Cast(torch.float32)
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.rmsnorm_mul = op.Multiply()
        if not self.with_scale:
            self.v_cast = VCast()

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype

        hidden_states = self.cast(hidden_states)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        if self.with_scale:
            return self.rmsnorm_mul(self.weight, hidden_states)
        else:
            out = self.rmsnorm_mul(self.weight, hidden_states)
            return self.v_cast(out)
        # if self.with_scale:
        #     return self.rmsnorm_mul(self.weight, hidden_states)
        # else:
        #     return self.rmsnorm_mul(hidden_states.new_tensor(self.weight_scalar), hidden_states)
        # return hidden_states


class Gemma4RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=500000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[0, 0, :, :]  # [seq_len, dim]
    sin = sin[0, 0, :, :]  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[0, 0, :, :]  # [seq_len, dim]
    sin = sin[0, 0, :, :]  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul_rr = op.Multiply()
        self.mul_ii = op.Multiply()
        self.sub = op.Subtract()
        self.mul_ri = op.Multiply()
        self.mul_ir = op.Multiply()
        self.add = op.Add()
        self.stack = op.Concat(3)

    def forward(self, x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
        x = self.apply_rope_single(x, rope_vals)
        return x

    def apply_rope_single(self, x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
        """
        Based on FacebookResearch's llama, provided by Carl
        """
        rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
        rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

        # TODO: Why HF uses different coordinates from the paper
        x_real = x[:, :, :, : x.shape[-1] // 2]  # extract first half elements
        x_im = x[:, :, :, x.shape[-1] // 2 :]  # extract second half elements

        x_prod_real = self.sub(
            self.mul_rr(x_real, rope_real), self.mul_ii(x_im, rope_im)
        )
        x_prod_im = self.add(self.mul_ri(x_real, rope_im), self.mul_ir(x_im, rope_real))

        # TODO: HF need to uses different interleaving
        x = self.stack(x_prod_real, x_prod_im).view(*x.shape).to(x.dtype)
        return x

class GeluPytorchTanh(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input):
        return torch.nn.functional.gelu(input)
        # return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

@QuantizationMixin.implements(GeluPytorchTanh)
class QuantizedGelu(QuantizationMixin, GeluPytorchTanh):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        # <TODO: Quantize `ret` as necessary>
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret

class VCast(nn.Module):
    def forward(self, x):
        return x

@QuantizationMixin.implements(VCast)
class QuantizedCast(_DispatchMixin, QuantizationMixin, VCast):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            ret = super().forward(x)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret

QuantizationMixin.ignore(Gemma4RotaryEmbedding)

class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    Quantized version of scaled word embedding.
    Multiplies embeddings by embed_scale using op.Multiply() for AIMET tracing.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.mul = op.Multiply()

    def forward(self, input: torch.Tensor):
        scale = self.weight.new_tensor(self.scalar_embed_scale)
        return self.mul(super().forward(input), scale)


class Gemma4TextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        use_double_wide_mlp: bool = False,
        is_kv_shared_layer: bool = False,
    ):
        super().__init__()
        # Apply double-wide MLP logic
        use_double_wide_mlp = use_double_wide_mlp and is_kv_shared_layer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size * (2 if use_double_wide_mlp else 1)

        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.act_fn = GeluPytorchTanh()
        self.mul = op.Multiply()


    def prepare_conv(self):
        if not hasattr(self, "forward_linear"):
            self.gate_proj_conv = nn.Conv2d(
                self.hidden_size, self.intermediate_size, 1, bias=False, 
                dtype=self.gate_proj.weight.dtype,
                device=self.gate_proj.weight.device
            )
            self.down_proj_conv = nn.Conv2d(
                self.intermediate_size, self.hidden_size, 1, bias=False, 
                dtype=self.down_proj.weight.dtype,
                device=self.down_proj.weight.device
            )
            self.up_proj_conv = nn.Conv2d(
                self.hidden_size, self.intermediate_size, 1, bias=False, 
                dtype=self.up_proj.weight.dtype,
                device=self.up_proj.weight.device
            )
            self.forward_linear = self.forward
            self.forward = self.forward_conv

        self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])
        self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])

        del self.gate_proj
        del self.down_proj
        del self.up_proj

    # considering generic lora
    def forward_conv(
        self,
        x,
        gate_lora_A_weights: Optional[Tuple[torch.Tensor]] = None,
        gate_lora_B_weights: Optional[Tuple[torch.Tensor]] = None,
        up_lora_A_weights: Optional[Tuple[torch.Tensor]] = None,
        up_lora_B_weights: Optional[Tuple[torch.Tensor]] = None,
        down_lora_A_weights: Optional[Tuple[torch.Tensor]] = None,
        down_lora_B_weights: Optional[Tuple[torch.Tensor]] = None,
    ):

        bsz, _, _ = x.size()

        x = torch.reshape(x, (bsz, -1, 1, self.hidden_size))
        x = x.transpose(1, 3)  # Transpose right before and after Conv

        gate_proj_out = self.gate_proj_conv(x)
        if gate_lora_A_weights is not None or up_lora_A_weights is not None or down_lora_A_weights is not None:
            raise NotImplementedError("Lora is not supported for MLP")
            

        up_proj_out = self.up_proj_conv(x)

        pre_down_proj = self.mul(self.act_fn(gate_proj_out), up_proj_out)

        x = self.down_proj_conv(pre_down_proj)
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.hidden_size))

        return x

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def repeat_kv(states, n_rep: int) -> List[torch.Tensor]:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    new_list = []
    for elem in states:
        for i in range(n_rep):
            new_list.append(elem)
    return new_list


class Gemma4TextAttention(nn.Module):
    """
    Quantized reimplementation of the reference Gemma4TextAttention.
    Uses per-head Conv2d projections (SHA), op.* elementwise wrappers,
    and always-on q/k/v norms. No LoRA, no flextron, no gated attention.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine layer type and sliding
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Head dimensions
        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and getattr(config, "global_head_dim", None)
            else config.head_dim
        )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # KV sharing — compute before creating projections
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        if self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            self.store_full_length_kv = (
                layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            ) if first_kv_shared_layer_idx > 0 else False

        # Projections — shared layers only need q_proj and o_proj (no k/v projections)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # Norms — q_norm always present; k/v norms only for non-shared layers
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        # RoPE
        rope_theta = getattr(config, "rope_theta", 500000)
        if self.is_sliding:
            rope_theta = getattr(config, "local_rope_theta", rope_theta)
        self.rotary_emb = Gemma4RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
            base=rope_theta,
        )

        # MHA ops (used before prepare_sha)
        self.matmul_qk = op.MatMul()
        self.matmul_sv = op.MatMul()
        self.q_rope_mha = RoPE()
        self.k_rope_mha = RoPE()

        self.mask_neg = getattr(config, "mask_neg", -100.0)
        self.return_new_key_value_only = getattr(config, "return_new_key_value_only", False)

    def prepare_sha(self):
        """Convert Linear projections to per-head Conv2d for AIMET quantization."""
        if hasattr(self, "forward_mha"):
            return  # already prepared

        del self.matmul_qk
        del self.matmul_sv

        # Per-head Conv2d projections — q and o always, k/v only for non-shared layers
        self.q_proj_sha = nn.ModuleList([
            nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False,
                      dtype=self.q_proj.weight.dtype,
                      device=self.q_proj.weight.device)
            for _ in range(self.num_heads)
        ])
        self.o_proj_conv = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=False,
                                     dtype=self.o_proj.weight.dtype,
                                     device=self.o_proj.weight.device)

        if not self.is_kv_shared_layer:
            self.k_proj_sha = nn.ModuleList([
                nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False,
                          dtype=self.k_proj.weight.dtype,
                          device=self.k_proj.weight.device)
                for _ in range(self.num_key_value_heads)
            ])
            self.v_proj_sha = nn.ModuleList([
                nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False,
                          dtype=self.v_proj.weight.dtype,
                          device=self.v_proj.weight.device)
                for _ in range(self.num_key_value_heads)
            ])

        # Per-head RoPE — q always, k only for non-shared layers
        self.q_rope = nn.ModuleList([RoPE() for _ in range(self.num_heads)])
        if not self.is_kv_shared_layer:
            self.k_rope = nn.ModuleList([RoPE() for _ in range(self.num_key_value_heads)])

        # KV cache concat ops — only for non-shared layers
        if not self.is_kv_shared_layer:
            self.k_cat = nn.ModuleList([op.Concat(3) for _ in range(self.num_key_value_heads)])
            self.v_cat = nn.ModuleList([op.Concat(2) for _ in range(self.num_key_value_heads)])

        # Per-head attention ops (always needed for attention computation)
        self.mm_qk = nn.ModuleList([op.MatMul() for _ in range(self.num_heads)])
        self.mask_add = nn.ModuleList([op.Add() for _ in range(self.num_heads)])
        self.div = nn.ModuleList([op.Divide() for _ in range(self.num_heads)])
        self.sm = nn.ModuleList([nn.Softmax(dim=-1) for _ in range(self.num_heads)])
        self.mm_qkv = nn.ModuleList([op.MatMul() for _ in range(self.num_heads)])
        self.o_cat = op.Concat(3)

        # Per-head norms for SHA — q always, k/v only for non-shared layers
        self.q_norm_sha = nn.ModuleList([
            Gemma4RMSNorm(self.head_dim, eps=self.q_norm.variance_epsilon, 
                          dtype=self.q_norm.weight.dtype,
                          device=self.q_norm.weight.device)
            for _ in range(self.num_heads)
        ])
        if not self.is_kv_shared_layer:
            self.k_norm_sha = nn.ModuleList([
                Gemma4RMSNorm(self.head_dim, eps=self.k_norm.variance_epsilon,
                              dtype=self.k_norm.weight.dtype,
                              device=self.k_norm.weight.device)
                for _ in range(self.num_key_value_heads)
            ])
            self.v_norm_sha = nn.ModuleList([
                Gemma4RMSNorm(self.head_dim, eps=self.v_norm.variance_epsilon, 
                              dtype=self.k_norm.weight.dtype,
                              device=self.k_norm.weight.device,
                              with_scale=False)
                for _ in range(self.num_key_value_heads)
            ])

        # Copy weights from Linear to Conv2d
        for i in range(self.num_heads):
            self.q_proj_sha[i].weight.data.copy_(
                self.q_proj.weight[i * self.head_dim : (i + 1) * self.head_dim, :, None, None]
            )
            # Copy q_norm weights to per-head norms
            self.q_norm_sha[i].weight.data.copy_(self.q_norm.weight.data)

        if not self.is_kv_shared_layer:
            for i in range(self.num_key_value_heads):
                self.k_proj_sha[i].weight.data.copy_(
                    self.k_proj.weight[i * self.head_dim : (i + 1) * self.head_dim, :, None, None]
                )
                self.v_proj_sha[i].weight.data.copy_(
                    self.v_proj.weight[i * self.head_dim : (i + 1) * self.head_dim, :, None, None]
                )
                self.k_norm_sha[i].weight.data.copy_(self.k_norm.weight.data)
                self.v_norm_sha[i].weight.data.copy_(self.v_norm.weight.data)

        self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

        # Delete original Linear projections
        del self.q_proj
        del self.q_norm
        del self.q_rope_mha
        del self.k_rope_mha
        if not self.is_kv_shared_layer:
            del self.k_proj
            del self.v_proj
            del self.k_norm
            del self.v_norm
        del self.o_proj

        # Switch forward method
        self.forward_mha = self.forward
        self.forward = self.forward_sha

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        shared_kv: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        return_new_key_value_only: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Select correct mask for sliding layers
        if self.is_sliding and sliding_attention_mask is not None:
            attention_mask = sliding_attention_mask

        bsz, q_len, _ = hidden_states.size()

        _shared_kv_cache = None

        # For KV shared layers, reuse pre-computed k/v
        if self.is_kv_shared_layer and shared_kv is not None:
            key_states, value_states = shared_kv
            # Still need to compute queries
            hidden_states_4d = hidden_states.reshape(bsz, -1, 1, self.hidden_size).transpose(1, 3)
            query_states = [
                q_proj(hidden_states_4d).permute(0, 2, 3, 1).view(bsz, 1, q_len, self.head_dim)
                for q_proj in self.q_proj_sha
            ]
            for i in range(self.num_heads):
                query_states[i] = self.q_norm_sha[i](query_states[i])

            # Apply RoPE to queries only
            if isinstance(position_ids, (tuple, list)):
                rope_embedding = position_ids
                if len(position_ids) == 4:
                    rope_embedding = position_ids[2:] if self.is_sliding else position_ids[:2]
                query_states = [
                    rope(q, rope_embedding) for rope, q in zip(self.q_rope, query_states)
                ]
            else:
                kv_seq_len = key_states[0].shape[-1]  # transposed keys
                cos, sin = self.rotary_emb(query_states[0], kv_seq_len)
                query_states = [
                    apply_rotary_pos_emb_single(q, cos, sin, position_ids)
                    for q in query_states
                ]

            # key_states and value_states are already computed (lists of tensors)
            present_key_value = None
        else:
            # Standard path: compute q, k, v
            hidden_states_4d = hidden_states.reshape(bsz, -1, 1, self.hidden_size).transpose(1, 3)

            query_states = [
                q_proj(hidden_states_4d).permute(0, 2, 3, 1).view(bsz, 1, q_len, self.head_dim)
                for q_proj in self.q_proj_sha
            ]
            key_states = [
                k_proj(hidden_states_4d).permute(0, 2, 3, 1).view(bsz, 1, q_len, self.head_dim)
                for k_proj in self.k_proj_sha
            ]
            value_states = [
                v_proj(hidden_states_4d).permute(0, 2, 3, 1).view(bsz, 1, q_len, self.head_dim)
                for v_proj in self.v_proj_sha
            ]

            # Apply norms
            for i in range(self.num_heads):
                query_states[i] = self.q_norm_sha[i](query_states[i])
            for i in range(self.num_key_value_heads):
                key_states[i] = self.k_norm_sha[i](key_states[i])
                value_states[i] = self.v_norm_sha[i](value_states[i])

            # RoPE
            kv_seq_len = value_states[0].shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[1][0].shape[-2]

            if isinstance(position_ids, (tuple, list)):
                rope_embedding = position_ids
                if len(position_ids) == 4:
                    rope_embedding = position_ids[2:] if self.is_sliding else position_ids[:2]
                query_states = [
                    rope(q, rope_embedding) for rope, q in zip(self.q_rope, query_states)
                ]
                key_states = [
                    rope(k, rope_embedding) for rope, k in zip(self.k_rope, key_states)
                ]
            else:
                cos, sin = self.rotary_emb(value_states[0], kv_seq_len)
                query_states = [
                    apply_rotary_pos_emb_single(q, cos, sin, position_ids)
                    for q in query_states
                ]
                key_states = [
                    apply_rotary_pos_emb_single(k, cos, sin, position_ids)
                    for k in key_states
                ]

            # Transpose keys for matmul: (bsz, 1, head_dim, seq_len)
            key_states = [k.transpose(2, 3) for k in key_states]

            if return_new_key_value_only:
                # Compilation mode: capture NEW kv only (before concat with past)
                present_key_value = (
                    (tuple(key_states), tuple(value_states)) if use_cache else None
                )

            # KV cache: concat past for attention computation
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key_states = [
                    cat(pk, k) for cat, pk, k in zip(self.k_cat, past_key, key_states)
                ]
                value_states = [
                    cat(pv, v) for cat, pv, v in zip(self.v_cat, past_value, value_states)
                ]

            # Store full KV for sharing to later shared layers
            if self.store_full_length_kv:
                _shared_kv_cache = (tuple(key_states), tuple(value_states))

            if not return_new_key_value_only:
                # Generate mode: return FULL concatenated kv
                present_key_value = (
                    (tuple(key_states), tuple(value_states)) if use_cache else None
                )

        # GQA repeat
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights — reference uses scaling=1.0 (no head_dim division)
        kv_seq_len = key_states[0].shape[-1]  # transposed: (bsz, 1, head_dim, kv_len)
        attn_weights = [
            mm(q, k)
            for mm, q, k in zip(self.mm_qk, query_states, key_states)
        ]

        # Apply mask
        if attention_mask is not None:
            attn_weights = [
                add(aw, attention_mask) for add, aw in zip(self.mask_add, attn_weights)
            ]

        # Softmax
        attn_weights = [
            sm(aw).to(query_states[0].dtype) for sm, aw in zip(self.sm, attn_weights)
        ]

        # Value multiply
        attn_output = [
            mm(aw, v) for mm, aw, v in zip(self.mm_qkv, attn_weights, value_states)
        ]

        # Concat heads and output projection
        attn_output = self.o_cat(*attn_output)
        attn_output = attn_output.permute(0, 3, 1, 2)
        attn_output = self.o_proj_conv(attn_output)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value, _shared_kv_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        shared_kv: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """MHA forward (before prepare_sha is called)."""
        if self.is_sliding and sliding_attention_mask is not None:
            attention_mask = sliding_attention_mask

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        if self.is_kv_shared_layer and shared_kv is not None:
            # Shared layers reuse KV from a source layer (QC tuple format)
            # Convert from tuple-of-per-head to batched MHA format
            keys_tuple, vals_tuple = shared_kv
            key_states = torch.cat([k.transpose(2, 3) for k in keys_tuple], dim=1)  # (bsz, kv_heads, seq, head_dim)
            value_states = torch.cat(list(vals_tuple), dim=1)  # (bsz, kv_heads, seq, head_dim)
        else:
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            key_states = self.k_norm(key_states)
            value_states = self.v_norm(value_states)

            # Transpose to (bsz, heads, seq, head_dim) for attention
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        query_states = query_states.transpose(1, 2)

        # RoPE — only apply to queries for shared layers (keys already have RoPE from source)
        if isinstance(position_ids, (tuple, list)):
            rope_embedding = position_ids
            if len(position_ids) == 4:
                rope_embedding = position_ids[2:] if self.is_sliding else position_ids[:2]
            query_states = self.q_rope_mha(query_states, rope_embedding).to(query_states.dtype)
            if not (self.is_kv_shared_layer and shared_kv is not None):
                key_states = self.k_rope_mha(key_states, rope_embedding).to(key_states.dtype)
        else:
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, kv_seq_len)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            if not (self.is_kv_shared_layer and shared_kv is not None):
                key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        # KV cache: concatenate past if present (skip for shared layers)
        if past_key_value is not None and not (self.is_kv_shared_layer and shared_kv is not None):
            past_keys, past_values = past_key_value
            # past_keys: tuple of (bsz, 1, head_dim, past_seq) per kv_head — transposed format
            # key_states: (bsz, kv_heads, new_seq, head_dim) — need to split, transpose, concat
            new_key_list = []
            new_val_list = []
            for h in range(self.num_key_value_heads):
                new_k = key_states[:, h:h+1, :, :].transpose(2, 3)  # (bsz, 1, head_dim, new_seq)
                new_v = value_states[:, h:h+1, :, :]  # (bsz, 1, new_seq, head_dim)
                new_key_list.append(torch.cat([past_keys[h], new_k], dim=3))
                new_val_list.append(torch.cat([past_values[h], new_v], dim=2))
            # Reconstruct (bsz, kv_heads, total_seq, head_dim)
            key_states = torch.cat([k.transpose(2, 3) for k in new_key_list], dim=1)
            value_states = torch.cat(new_val_list, dim=1)

        # Store full KV for sharing to later shared layers
        if self.store_full_length_kv:
            kv_key_list_shared = []
            kv_val_list_shared = []
            for h in range(self.num_key_value_heads):
                kv_key_list_shared.append(key_states[:, h:h+1, :, :].transpose(2, 3))
                kv_val_list_shared.append(value_states[:, h:h+1, :, :])
            _shared_kv_cache = (tuple(kv_key_list_shared), tuple(kv_val_list_shared))
        else:
            _shared_kv_cache = None

        # KV cache — store per-kv-head as QC-style tuple-of-tuples (skip for shared layers)
        if use_cache and not (self.is_kv_shared_layer and shared_kv is not None):
            kv_key_list = []
            kv_val_list = []
            for h in range(self.num_key_value_heads):
                kv_key_list.append(key_states[:, h:h+1, :, :].transpose(2, 3))  # (bsz, 1, head_dim, seq)
                kv_val_list.append(value_states[:, h:h+1, :, :])  # (bsz, 1, seq, head_dim)
            present_key_value = (tuple(kv_key_list), tuple(kv_val_list))
        else:
            present_key_value = None

        # Attention — reference uses scaling=1.0 (no head_dim division)
        # GQA repeat
        key_states_rep = key_states.repeat(1, self.num_key_value_groups, 1, 1) if self.num_key_value_groups > 1 else key_states
        value_states_rep = value_states.repeat(1, self.num_key_value_groups, 1, 1) if self.num_key_value_groups > 1 else value_states

        attn_weights = self.matmul_qk(query_states, key_states_rep.transpose(2, 3))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = self.matmul_sv(attn_weights, value_states_rep)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value, _shared_kv_cache 


class Gemma4TextDecoderLayer(nn.Module):
    """
    Quantized reimplementation of the reference Gemma4TextDecoderLayer.
    Uses 4-norm sandwich (input/post-attn/pre-ff/post-ff), elementwise ops,
    PLE block with gelu_pytorch_tanh. No MoE.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)

        # MLP — reuse existing quantized MLP
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        self.mlp = Gemma4TextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_activation", "gelu_pytorch_tanh"),
            use_double_wide_mlp=getattr(config, "use_double_wide_mlp", False),
            is_kv_shared_layer=is_kv_shared_layer,
        )

        # 4 norms (matching reference)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Layer scalar buffer
        self.register_buffer("layer_scalar", torch.ones(1))

        # Elementwise ops for residual connections
        self.add_attn = op.Add()
        self.add_mlp = op.Add()
        self.mul_scalar = op.Multiply()

        # Per-layer input (PLE) block
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            self.act_fn = GeluPytorchTanh()
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.mul_ple = op.Multiply()
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.add_ple = op.Add()

    def prepare_conv(self):
        """Convert PLE linear layers to Conv2d and switch to conv forward."""
        if self.hidden_size_per_layer_input and not hasattr(self, "forward_linear"):
            self.per_layer_input_gate_conv = nn.Conv2d(
                self.hidden_size, self.hidden_size_per_layer_input, 1, bias=False,
                dtype=self.per_layer_input_gate.weight.dtype,
                device = self.per_layer_input_gate.weight.device
            )
            self.per_layer_projection_conv = nn.Conv2d(
                self.hidden_size_per_layer_input, self.hidden_size, 1, bias=False,
                dtype=self.per_layer_projection.weight.dtype,
                device = self.per_layer_projection.weight.device
            )
            self.forward_linear = self.forward
            self.forward = self.forward_conv

        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate_conv.weight.data.copy_(
                self.per_layer_input_gate.weight[:, :, None, None]
            )
            self.per_layer_projection_conv.weight.data.copy_(
                self.per_layer_projection.weight[:, :, None, None]
            )
            del self.per_layer_input_gate
            del self.per_layer_projection

    def _decoder_body(self, hidden_states, present_key_value, per_layer_input, use_ple_conv=False, use_cache=False, residual = None):
        """Shared post-attention body for both linear and conv forward paths."""
        # 2. Post-attention norm + residual
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.add_attn(residual, hidden_states)

        # 3. Pre-feedforward norm + MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # 4. Post-feedforward norm + residual
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.add_mlp(residual, hidden_states)

        # 5. Per-layer input block
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            if use_ple_conv:
                bsz, q_len, _ = hidden_states.size()
                x = hidden_states.reshape(bsz, -1, 1, self.hidden_size).transpose(1, 3)
                x = self.per_layer_input_gate_conv(x)
                x = x.transpose(1, 3).reshape(bsz, q_len, self.hidden_size_per_layer_input)
                x = self.act_fn(x)
                x = self.mul_ple(x, per_layer_input)
                x = x.reshape(bsz, -1, 1, self.hidden_size_per_layer_input).transpose(1, 3)
                x = self.per_layer_projection_conv(x)
                x = x.transpose(1, 3).reshape(bsz, q_len, self.hidden_size)
            else:
                x = self.per_layer_input_gate(hidden_states)
                x = self.act_fn(x)
                x = self.mul_ple(x, per_layer_input)
                x = self.per_layer_projection(x)
            hidden_states = self.post_per_layer_input_norm(x)
            hidden_states = self.add_ple(residual, hidden_states)

        # 6. Layer scalar
        hidden_states = self.mul_scalar(hidden_states, self.layer_scalar)

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states

    def forward_conv(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        shared_kv: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        return_new_key_value_only: bool = True,
    ):
        """Conv2d forward path. return_new_key_value_only=True for compilation, False for generate."""
        _residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, _shared_kv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            shared_kv=shared_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_new_key_value_only=return_new_key_value_only,
        )
        layer_output = self._decoder_body(hidden_states, present_key_value if use_cache else None, per_layer_input, use_ple_conv=True, use_cache=use_cache, residual=_residual)
        return (layer_output, _shared_kv_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        shared_kv: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        return_new_key_value_only: bool = False,
    ):
        """Linear forward path (generate mode) — returns full concatenated KV by default."""
        _residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, _shared_kv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            shared_kv=shared_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        layer_output = self._decoder_body(hidden_states, present_key_value if use_cache else None, per_layer_input, use_ple_conv=False, use_cache=use_cache, residual=_residual)
        return layer_output, _shared_kv_cache


class Gemma4PreTrainedModel(PreTrainedModel):
    """Base class for QC Gemma4 text models."""
    config_class = Gemma4Config
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4TextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)


class Gemma4TextModel(Gemma4PreTrainedModel):
    """
    Quantized reimplementation of the reference Gemma4TextModel.
    Embeddings are external (use_input_embedding_input=True).
    Receives inputs_embeds and per_layer_inputs as pre-computed tensors.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.is_folded = getattr(config, "is_folded", False)

        # Embedding layers (for from_pretrained loading under model.* prefix)
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size ** 0.5 if not self.is_folded else 1.0
        )

        if self.hidden_size_per_layer_input:
            vocab_size_per_layer_input = getattr(config, "vocab_size_per_layer_input", config.vocab_size)
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                vocab_size_per_layer_input,
                self.num_hidden_layers * self.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=self.hidden_size_per_layer_input ** 0.5 if not self.is_folded else 1.0
            )
            self.per_layer_input_scale = 2.0 ** -0.5 if not self.is_folded else 1.0
            if not self.is_folded:
                self.per_layer_model_projection = nn.Linear(
                    config.hidden_size,
                    self.num_hidden_layers * self.hidden_size_per_layer_input,
                    bias=False,
                )
                self.per_layer_model_projection_scale = config.hidden_size ** -0.5
                self.per_layer_projection_norm = Gemma4RMSNorm(
                    self.hidden_size_per_layer_input,
                    eps=config.rms_norm_eps,
                )

        # Decoder layers
        self.layers = nn.ModuleList([
            Gemma4TextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, mask_neg=-100.0):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                mask_neg=mask_neg,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1], mask_neg=mask_neg).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    @staticmethod
    def _prepare_decoder_sliding_attention_mask(attention_mask, sliding_window, input_shape, inputs_embeds, past_key_values_length, mask_neg=-100.0):
        ## TODO(goka): implement sliding attention mask
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                mask_neg=mask_neg,
            )
            slw_mask = torch.tril(torch.ones_like(combined_attention_mask, dtype=torch.bool), diagonal=-sliding_window)
            combined_attention_mask = torch.where(slw_mask, mask_neg, combined_attention_mask)
        else:
            swa = attention_mask.clone()
            swa[:, :-sliding_window] = 0
            combined_attention_mask = _expand_mask(swa, inputs_embeds.dtype, tgt_len=input_shape[-1], mask_neg=mask_neg).to(
                inputs_embeds.device
            )
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len] 
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1], mask_neg=mask_neg).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @torch.no_grad()
    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
        is_folded: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        """
        Convert input_ids to inputs_embeds, per_layer_inputs, and 4D attention masks.
        This is NOT called inside forward() — call it separately before forward.

        Args:
            input_ids: (batch, seq)
            attention_mask: 2D (batch, total_seq_len) with 1=attend/0=pad, or None.
            past_key_values_length: length of past KV cache for mask sizing.

        Returns:
            inputs_embeds: (batch, seq, hidden_size)
            per_layer_inputs: list of (batch, seq, hidden_size_per_layer_input), or None.
            attention_mask_4d: (batch, 1, seq, seq + past_len) causal mask for full attention layers.
            sliding_attention_mask_4d: (batch, 1, seq, seq + past_len) sliding window mask.
        """
        inputs_embeds = self.embed_tokens(input_ids)

        per_layer_inputs_list = None
        if self.hidden_size_per_layer_input:
            raw_per_layer = self.embed_tokens_per_layer(input_ids).reshape(
                *input_ids.shape,
                self.num_hidden_layers,
                self.hidden_size_per_layer_input,
            )
            if is_folded:
                per_layer_inputs_tensor = raw_per_layer
            else:
                per_layer_projection = (
                    self.per_layer_model_projection(inputs_embeds)
                    * self.per_layer_model_projection_scale
                )
                per_layer_projection = per_layer_projection.reshape(
                    *inputs_embeds.shape[:-1],
                    self.num_hidden_layers,
                    self.hidden_size_per_layer_input,
                )
                per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
                projected = (per_layer_projection + raw_per_layer) * self.per_layer_input_scale
                per_layer_inputs_tensor = projected
            per_layer_inputs_list = [
                per_layer_inputs_tensor[:, :, i, :] for i in range(self.num_hidden_layers)
            ]

        # Create 4D attention masks
        input_shape = input_ids.shape
        mask_neg = getattr(self.config, "mask_neg", -100.0)
        sliding_window = getattr(self.config, "sliding_window", 512)

        attention_mask_4d = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds,
            past_key_values_length, mask_neg=mask_neg,
        )
        sliding_attention_mask_4d = self._prepare_decoder_sliding_attention_mask(
            attention_mask, sliding_window, input_shape, inputs_embeds,
            past_key_values_length, mask_neg=mask_neg,
        )

        return inputs_embeds, per_layer_inputs_list, attention_mask_4d, sliding_attention_mask_4d

    def prepare_sha(self):
        """Prepare all layers for SHA (Conv2d) mode."""
        for layer in self.layers:
            layer.self_attn.prepare_sha()
            layer.mlp.prepare_conv()
            layer.prepare_conv()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_new_key_value_only: bool = True,
        skip_shared_layers: bool = False,
    ) -> BaseModelOutputWithPast:
        """
        Args:
            inputs_embeds: Pre-computed embeddings (batch, seq, hidden_size).
            attention_mask: Full attention mask (batch, 1, q_len, kv_len) or None.
            sliding_attention_mask: Sliding window mask or None.
            position_ids: RoPE as 4-tuple (global_cos, global_sin, sliding_cos, sliding_sin)
                          from RopeEmbedding, or integer position_ids.
            per_layer_inputs: List of (batch, seq, hidden_size_per_layer_input), one per layer.
            past_key_values: List of (keys_tuple, values_tuple) per layer, or None.
            use_cache: Whether to return KV cache.
            return_new_key_value_only: If True (compilation), return only new KV.
                                       If False (generate), return full concatenated KV.
        """
        hidden_states = inputs_embeds

        # Handle HF DynamicCache: convert to list or treat as None
        if past_key_values is not None and not isinstance(past_key_values, list):
            # HF generate passes DynamicCache — ignore it, we manage cache internally
            past_key_values = None

        # Auto-create 4D masks if not provided (fallback for callers that skip prepare_inputs)
        if attention_mask is None or attention_mask.dim() == 2:
            bsz, seq_len = inputs_embeds.shape[:2]
            input_shape = (bsz, seq_len)
            past_key_values_length = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_key_values_length = past_key_values[0][0][0].shape[-1]
            mask_neg = getattr(self.config, "mask_neg", -100.0)
            sliding_window = getattr(self.config, "sliding_window", 512)
            attention_mask_2d = attention_mask  # None or (batch, total_seq)
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask_2d, input_shape, inputs_embeds,
                past_key_values_length, mask_neg=mask_neg,
            )
            sliding_attention_mask = self._prepare_decoder_sliding_attention_mask(
                attention_mask_2d, sliding_window, input_shape, inputs_embeds,
                past_key_values_length, mask_neg=mask_neg,
            )

        all_present_key_values = () if use_cache else None
        shared_kv_store = {}  # layer_idx -> (keys_tuple, values_tuple) for KV sharing
        past_kv_idx = 0

        for i, decoder_layer in enumerate(self.layers):
            attn = decoder_layer.self_attn

            if skip_shared_layers and attn.is_kv_shared_layer:
                continue
            per_layer_input = per_layer_inputs[i] if per_layer_inputs is not None else None

            past_kv = None
            if past_key_values is not None and not attn.is_kv_shared_layer:
                if past_kv_idx < len(past_key_values):
                    past_kv = past_key_values[past_kv_idx]
                past_kv_idx += 1

            # For KV-shared layers, look up shared KV from the source layer
            shared_kv = None
            if attn.is_kv_shared_layer:
                shared_kv = shared_kv_store.get(attn.kv_shared_layer_index)

            layer_output, _shared_kv_cache = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                attention_mask=attention_mask,
                sliding_attention_mask=sliding_attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                shared_kv=shared_kv,
                use_cache=use_cache,
                return_new_key_value_only=return_new_key_value_only,
            )

            if use_cache:
                hidden_states, present_kv = layer_output
                if present_kv is not None:
                    all_present_key_values += (present_kv,)
            else:
                hidden_states = layer_output

            # Store full KV from source layers for sharing
            if attn.store_full_length_kv and _shared_kv_cache is not None:
                shared_kv_store[i] = _shared_kv_cache

        if not skip_shared_layers:
            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=all_present_key_values if use_cache else None,
        )


class Gemma4TextForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    """
    Quantized reimplementation of the reference Gemma4ForCausalLM.
    Inherits from PreTrainedModel + GenerationMixin for checkpoint loading and HF generate().
    Embeddings are external (computed in Gemma4TextInputPreparer set via set_input_preparer).
    Includes lm_head with Conv2d support and logit softcapping.
    """
    _tied_weights_keys = {}

    def __init__(self, config, return_kv_cache_only: bool = False):
        super().__init__(config)
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.return_kv_cache_only = return_kv_cache_only
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE function — set via set_rope_fn() for generate()
        self._rope_fn = None

        # Softcapping ops
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        if self.final_logit_softcapping is not None:
            self.div_softcap = op.Divide()
            self.tanh_softcap = nn.Tanh()
            self.mul_softcap = op.Multiply()
            self._softcap_float = float(self.final_logit_softcapping)

        self.post_init()

    def set_rope_fn(self, rope_fn):
        """
        Set the RoPE function for generate().
        Must be called before using HF generate().

        Args:
            rope_fn: Callable(position_ids) -> 4-tuple RoPE
                     (global_cos, global_sin, sliding_cos, sliding_sin).
        """
        self._rope_fn = rope_fn

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
        is_folded: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        """
        Convert input_ids to inputs_embeds, per_layer_inputs, and 4D attention masks.

        Args:
            input_ids: (batch, seq)
            attention_mask: 2D (batch, total_seq_len) with 1=attend/0=pad, or None.
            past_key_values_length: length of past KV cache for mask sizing.

        Returns:
            inputs_embeds, per_layer_inputs, attention_mask_4d, sliding_attention_mask_4d
        """
        return self.model.prepare_inputs(input_ids, attention_mask, past_key_values_length, is_folded)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """
        Override HF GenerationMixin to:
        1. Convert input_ids -> inputs_embeds + per_layer_inputs via input_preparer
        2. Compute RoPE 4-tuple via rope_fn
        3. Pass our QC-style past_key_values (list of tuples) through unchanged
        4. On subsequent steps (past_key_values present), only process the last token
        """
        # Determine past KV length and slice input_ids for cached generation
        past_len = 0
        if past_key_values is not None and isinstance(past_key_values, list):
            # past exists and is our QC format — slice to last token only
            past_len = past_key_values[0][0][0].shape[-1]
            input_ids = input_ids[:, -1:]

        # Ignore DynamicCache from HF — only pass our list-of-tuples
        if past_key_values is not None and not isinstance(past_key_values, list):
            past_key_values = None

        # Convert input_ids to embeddings, per_layer_inputs, and 4D masks
        # attention_mask from HF is 2D (batch, total_seq) with 1=attend/0=pad
        inputs_embeds, per_layer_inputs, attention_mask_4d, sliding_attention_mask_4d = \
            self.model.prepare_inputs(input_ids, attention_mask, past_key_values_length=past_len,
                                      is_folded=self.model.is_folded)

        # Compute position_ids and RoPE
        if past_len > 0:
            position_ids = torch.arange(
                past_len, past_len + input_ids.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
        else:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device,
            ).unsqueeze(0)

        if self._rope_fn is not None:
            position_ids = self._rope_fn(position_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "per_layer_inputs": per_layer_inputs,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask_4d,
            "sliding_attention_mask": sliding_attention_mask_4d,
            "use_cache": use_cache,
            "return_new_key_value_only": False,
        }

    def prepare_conv(self):
        """Convert lm_head Linear to Conv2d and switch to conv forward."""
        if not hasattr(self, "forward_linear"):
            self.lm_head_conv = nn.Conv2d(
                self.config.hidden_size, self.config.vocab_size, 1, bias=False,
                dtype = self.lm_head.weight.dtype,
                device = self.lm_head.weight.device
            )
            self.forward_linear = self.forward
            self.forward = self.forward_conv

        self.lm_head_conv.weight.data.copy_(self.lm_head.weight[:, :, None, None])
        del self.lm_head

    def enable_conv(self):
        """Convert all submodules to Conv2d mode."""
        self.prepare_conv()
        self.model.prepare_sha()

    def _apply_softcapping(self, logits):
        """Apply logit softcapping if configured."""
        if self.final_logit_softcapping is not None:
            softcap = logits.new_tensor(self._softcap_float)
            logits = self.div_softcap(logits, softcap)
            logits = self.tanh_softcap(logits)
            logits = self.mul_softcap(logits, softcap)
        return logits

    def forward_conv(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        return_new_key_value_only: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Conv2d forward path. return_new_key_value_only=True for compilation, False for generate."""
        if position_ids is not None and not isinstance(position_ids, (tuple, list)):
            if self._rope_fn is not None:
                position_ids = self._rope_fn(position_ids)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            position_ids=position_ids,
            per_layer_inputs=per_layer_inputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_new_key_value_only=return_new_key_value_only,
            skip_shared_layers=self.return_kv_cache_only,
        )

        if self.return_kv_cache_only:
            if not self.config.return_dict:
                return outputs.past_key_values
            return CausalLMOutputWithPast(
                logits=None,
                past_key_values=outputs.past_key_values,
            )

        hidden_states = outputs.last_hidden_state

        # Slice logits if needed
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        x = hidden_states[:, slice_indices, :]

        # lm_head via Conv2d: (b, l, h) -> (b, h, 1, l) -> conv -> (b, v, 1, l) -> (b, l, v)
        bsz, seq_len, _ = x.size()
        x = x.reshape(bsz, -1, 1, self.config.hidden_size).transpose(1, 3)
        x = self.lm_head_conv(x)
        logits = x.transpose(1, 3).reshape(bsz, seq_len, self.config.vocab_size)

        logits = self._apply_softcapping(logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not self.config.return_dict:
            return (logits, outputs.past_key_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        return_new_key_value_only: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Linear forward path (default, before prepare_conv is called).
        Accepts pre-computed inputs_embeds and per_layer_inputs.
        Use model.prepare_inputs(input_ids) to convert input_ids before calling forward.
        HF generate() handles this via prepare_inputs_for_generation().
        """
        # Compute RoPE if position_ids are integer (not pre-computed 4-tuple)
        if position_ids is not None and not isinstance(position_ids, (tuple, list)):
            if self._rope_fn is not None:
                position_ids = self._rope_fn(position_ids)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            position_ids=position_ids,
            per_layer_inputs=per_layer_inputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_new_key_value_only=return_new_key_value_only,
            skip_shared_layers=self.return_kv_cache_only,
        )

        if self.return_kv_cache_only:
            if not self.config.return_dict:
                return outputs.past_key_values
            return CausalLMOutputWithPast(
                logits=None,
                past_key_values=outputs.past_key_values,
            )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        logits = self._apply_softcapping(logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not self.config.return_dict:
            return (logits, outputs.past_key_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

