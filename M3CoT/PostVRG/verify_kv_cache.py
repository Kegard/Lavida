"""Verify KV cache correctness by comparing logits from 3 forward modes."""
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from M3CoT.PostVRG.postvrg import MASK_TOKEN_ID


def compare_forward_modes(core_model, prefix_embeds, max_new_tokens=64):
    """Compare logits from 3 forward modes on the same input."""
    device = prefix_embeds.device
    prefix_length = prefix_embeds.shape[1]
    seq_len = prefix_length + max_new_tokens

    x = torch.full((1, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0

    # --- Mode 1: full-sequence forward (old baseline) ---
    full_embeds = core_model.transformer.wte(x)
    full_embeds[:, :prefix_length] = prefix_embeds
    logits_full = core_model(None, input_embeddings=full_embeds).logits

    # --- Mode 2: KV cache forward (new code) ---
    # step a: cache prefix
    prefix_out = core_model(None, input_embeddings=prefix_embeds, use_cache=True)
    prefix_kv = prefix_out.attn_key_values
    # step b: forward answer with cache
    answer_embeds = core_model.transformer.wte(x[:, prefix_length:])
    answer_out = core_model(None, input_embeddings=answer_embeds, past_key_values=prefix_kv)
    logits_kv = answer_out.logits  # shape: (1, max_new_tokens, vocab)

    # --- Mode 3: llada_generate style (prefix_lm=True) ---
    # same as mode 2 conceptually, but uses the same exact API as generate.py
    prefix_out2 = core_model(None, input_embeddings=prefix_embeds, use_cache=True)
    prefix_kv2 = prefix_out2.attn_key_values
    answer_embeds2 = core_model.transformer.wte(x[:, prefix_length:])
    answer_out2 = core_model(None, input_embeddings=answer_embeds2, past_key_values=prefix_kv2)
    logits_gen = answer_out2.logits

    # --- Compare ---
    answer_logits_full = logits_full[:, prefix_length:]  # (1, max_new_tokens, vocab)
    answer_logits_kv = logits_kv                          # (1, max_new_tokens, vocab)
    answer_logits_gen = logits_gen                         # (1, max_new_tokens, vocab)

    # KV cache vs generate (should be identical)
    diff_kv_gen = (answer_logits_kv - answer_logits_gen).abs()
    # full vs KV cache (expected to differ)
    diff_full_kv = (answer_logits_full - answer_logits_kv).abs()

    # token predictions
    pred_full = answer_logits_full.argmax(dim=-1)[0]
    pred_kv = answer_logits_kv.argmax(dim=-1)[0]
    pred_gen = answer_logits_gen.argmax(dim=-1)[0]

    token_match_kv_gen = (pred_kv == pred_gen).float().mean().item()
    token_match_full_kv = (pred_full == pred_kv).float().mean().item()

    print(f"prefix_length={prefix_length}, answer_length={max_new_tokens}")
    print()
    print("KV-cache vs llada_generate (should be ~0):")
    print(f"  max diff: {diff_kv_gen.max().item():.6e}")
    print(f"  mean diff: {diff_kv_gen.mean().item():.6e}")
    print(f"  token match: {token_match_kv_gen:.4f}")
    print()
    print("Full-sequence vs KV-cache (expected to differ):")
    print(f"  max diff: {diff_full_kv.max().item():.6e}")
    print(f"  mean diff: {diff_full_kv.mean().item():.6e}")
    print(f"  token match: {token_match_full_kv:.4f}")
    print()

    # show first 10 token predictions
    print("First 10 answer token predictions:")
    print(f"  full:     {pred_full[:10].tolist()}")
    print(f"  kv_cache: {pred_kv[:10].tolist()}")
    print(f"  generate: {pred_gen[:10].tolist()}")

    return {
        "kv_gen_max_diff": diff_kv_gen.max().item(),
        "full_kv_max_diff": diff_full_kv.max().item(),
        "token_match_kv_gen": token_match_kv_gen,
        "token_match_full_kv": token_match_full_kv,
    }


def main():
    import argparse
    import datasets
    from Scale_Attention.reweight_patch import get_torch_dtype, maybe_disable_torch_compile
    from M3CoT.run_m3cot_stepwise_x0 import prepare_prefix

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    # minimal args object for prepare_prefix
    class Args:
        pretrained = args.pretrained
        model_name = "llava_llada"
        device_map = "auto"
        torch_dtype = "bfloat16"
        vision_tower = str(REPO_ROOT / "weight" / "siglip")
        prompt = "cot"
        conv_template = "llada"
        device = "cuda"

    pargs = Args()
    restore_compile = maybe_disable_torch_compile()

    from llava.model.builder import load_pretrained_model
    vision_kwargs = dict(
        mm_vision_tower=pargs.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        mm_pooler_ratio=2,
        mm_patch_merge_type="spatial_unpad",
        use_mm_proj=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pargs.pretrained, None, pargs.model_name,
        device_map=pargs.device_map, vision_kwargs=vision_kwargs,
        torch_dtype=pargs.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(pargs.torch_dtype))
    core_model = model.get_model()

    dataset = datasets.load_dataset("LightChen2333/M3CoT", split="test")
    dataset = dataset.shuffle(seed=42)

    for i, doc in enumerate(dataset):
        if i >= args.num_samples:
            break
        if doc.get("image") is None:
            continue
        print(f"\n{'='*60}")
        print(f"Sample {i}: {doc['id']}")
        print(f"{'='*60}")
        _, _, prefix_embeds, _ = prepare_prefix(pargs, model, tokenizer, image_processor, doc)
        with torch.no_grad():
            compare_forward_modes(core_model, prefix_embeds, args.max_new_tokens)

    if restore_compile:
        restore_compile()


if __name__ == "__main__":
    main()
