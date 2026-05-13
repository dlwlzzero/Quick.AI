import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoProcessor
from gemma4_qc.gemma4_text import Gemma4TextForCausalLM
import math
from typing import Optional

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace repo id or local path to the original Gemma4 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the baked checkpoint.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Tokens processed per forward call (memory/speed tradeoff).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for preprocessing (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        default = False,
        help="Re-run prepare_inputs on a random sample and assert parity.",
    )
    return parser.parse_args()


def verify_parity(reference, converted, sample_size=64, seed=123456789):
    config = reference.model.config
    device = reference.device
    vocab = getattr(config, "vocab_size_per_layer_input", config.vocab_size)
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, sample_size), generator=generator)

    ref_embeds, ref_list, _, _ = reference.model.prepare_inputs(ids.to(device), is_folded=False)
    folded_embeds, folded_ple_list, _, _ = converted.model.prepare_inputs(ids.to(device), is_folded=True)

    if ref_list is None or folded_ple_list is None:
        raise RuntimeError("per_layer_inputs missing from one of the models")

    max_abs = (ref_embeds.float() - folded_embeds.float()).abs().max().item()
    if max_abs > 1e-1:
        raise AssertionError(
            f"Layer embeddings: max abs diff {max_abs:.3e} exceeds tolerance"
        )

    for layer_idx, (ref, bak) in enumerate(zip(ref_list, folded_ple_list)):
        max_abs = (ref.float() - bak.float()).abs().max().item()
        if max_abs > 5e-1:
            raise AssertionError(
                f"layer {layer_idx}: max abs diff {max_abs:.3e} exceeds tolerance"
            )
    print(f"verify: parity OK on {sample_size} random tokens")

def apply_fold(model, folded_ple):
    text_model = model.model
    target = text_model.embed_tokens_per_layer.weight

    if folded_ple.shape != target.shape:
        raise ValueError(
            f"Folded weight shape {tuple(folded_ple.shape)} does not match "
            f"embed_tokens_per_layer.weight shape {tuple(target.shape)}"
        )

    target.data.copy_(folded_ple.to(device=target.device, dtype=target.dtype))

    # Zero projection: per_layer_model_projection(inputs_embeds) -> 0.
    text_model.per_layer_model_projection.weight.data.zero_()
    # per_layer_projection_norm.weight is left untouched; RMSNorm(0) = 0 regardless.
    
    text_model.embed_tokens.weight.data.copy_(text_model.embed_tokens.weight * text_model.embed_tokens.scalar_embed_scale)
    text_model.embed_tokens.scalar_embed_scale = 1.0
    text_model.embed_tokens_per_layer.scalar_embed_scale = 1.0
    model.config.is_folded = True


def compute_folded_per_layer(model, chunk_size, compute_dtype, device):
    text_model = model.model
    config = text_model.config

    if not text_model.hidden_size_per_layer_input:
        raise ValueError("Model has no per-layer embedding path; nothing to bake.")

    vocab_per_layer: int = getattr(
        config, "vocab_size_per_layer_input", config.vocab_size
    )
    num_layers: int = text_model.num_hidden_layers
    h_pl: int = text_model.hidden_size_per_layer_input

    model.eval()
    # Cast just the per-layer preprocessing submodules to compute_dtype.
    # Leaves the rest of the model untouched.
    for name in (
        "embed_tokens",
        "embed_tokens_per_layer",
        "per_layer_model_projection",
        "per_layer_projection_norm",
    ):
        getattr(text_model, name).to(device=device, dtype=compute_dtype)

    folded_ple = torch.empty(
        vocab_per_layer,
        num_layers * h_pl,
        dtype=compute_dtype,
        device=device,
    )

    for start in range(0, vocab_per_layer, chunk_size):
        end = min(start + chunk_size, vocab_per_layer)
        ids = torch.arange(start, end, device=device).unsqueeze(0)  # (1, chunk)
        # prepare_inputs returns projected per-layer list; stack into (chunk, L, H_pl).
        _, per_layer_list, _, _ = text_model.prepare_inputs(
            ids, attention_mask=None, past_key_values_length=0, is_folded=False
        )
        stacked = torch.stack(per_layer_list, dim=2).squeeze(0)  # (chunk, L, H_pl)
        folded_ple[start:end] = stacked.reshape(end - start, num_layers * h_pl)

    return folded_ple



def main() -> None:
    args = parse_args()
    compute_dtype = torch.float32
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model from {args.model_id}...")
    model = Gemma4TextForCausalLM.from_pretrained(args.model_id)
    model.to(device)

    if not args.no_verify:
        # Keep a frozen copy of the original per-layer submodules for comparison.
        print("cloning model for verification...")
        reference = Gemma4TextForCausalLM.from_pretrained(args.model_id).to(device)
    else:
        reference: Optional[Gemma4TextForCausalLM] = None

    print(
        f"folding per-layer embeddings "
        f"(chunk_size={args.chunk_size}, dtype={torch.float32})..."
    )
    folded_ple = compute_folded_per_layer(
        model=model,
        chunk_size=args.chunk_size,
        compute_dtype=compute_dtype,
        device=device,
    )

    print("applying folded weight and neutralising projection path...")
    apply_fold(model, folded_ple)

    if reference is not None:
        verify_parity(reference, model)

    print(f"saving folded checkpoint to {output_dir}...")
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(str(output_dir))
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.save_pretrained(str(output_dir))

    print("done.")


if __name__ == "__main__":
    main()
