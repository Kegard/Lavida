import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import IGNORE_INDEX
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch

from Attention.analyze_step_unmask_attention import (
    CATEGORY_ORDER,
    GEN_TEXT_CATEGORY_ORDER,
    MASK_TOKEN_ID,
    QueryAttentionCollector,
    analyze_single_example,
    average_query_stats,
    build_prompt,
    get_special_token_ids,
    get_torch_dtype,
    load_pretrained_model,
    prepare_multimodal_prefix,
    save_attention_distribution_plot,
    save_gen_text_attention_distribution_plot,
    save_layer_step_heatmap,
    save_step_attention_distribution_plot,
    save_visual_attention_curve,
    save_visual_attention_over_steps,
)
from Scale_Attention.reweight_patch import patch_category_reweight_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and reweighted LaViDa attention distributions.")
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default="images/dog.png")
    parser.add_argument("--question", default="Describe the image in detail.")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--schedule", default="none")
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--output-dir", default="Scale_Attention/outputs/reweight_attention_compare")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--alpha-prompt", type=float, default=1.0)
    parser.add_argument("--alpha-visual", type=float, default=1.0)
    parser.add_argument("--alpha-generated", type=float, default=1.0)
    parser.add_argument("--alpha-mask", type=float, default=None)
    parser.add_argument("--alpha-normal", type=float, default=None)
    parser.add_argument("--alpha-special", type=float, default=None)
    return parser.parse_args()


def build_reweight_weights(prefix_ids, visual_mask, gen_tokens, special_token_ids, args):
    alpha_mask = float(args.alpha_generated if args.alpha_mask is None else args.alpha_mask)
    alpha_normal = float(args.alpha_generated if args.alpha_normal is None else args.alpha_normal)
    alpha_special = float(args.alpha_generated if args.alpha_special is None else args.alpha_special)

    prefix_ids = prefix_ids.to(dtype=torch.long)
    visual_mask = visual_mask.to(device=prefix_ids.device, dtype=torch.bool)
    prompt_mask = (~visual_mask) & (prefix_ids != IGNORE_INDEX)

    gen_tokens = gen_tokens.to(device=prefix_ids.device, dtype=torch.long).view(-1)
    generated_is_mask = gen_tokens == MASK_TOKEN_ID
    generated_is_special = torch.zeros_like(generated_is_mask)
    special_token_ids = [token_id for token_id in special_token_ids if token_id != MASK_TOKEN_ID]
    if special_token_ids:
        special_ids = torch.tensor(special_token_ids, dtype=torch.long, device=gen_tokens.device)
        generated_is_special = torch.isin(gen_tokens, special_ids) & (~generated_is_mask)
    generated_is_normal = (~generated_is_mask) & (~generated_is_special)

    weights = torch.ones(prefix_ids.shape[0] + gen_tokens.shape[0], dtype=torch.float32, device=prefix_ids.device)
    weights[: prefix_ids.shape[0]][visual_mask] = float(args.alpha_visual)
    weights[: prefix_ids.shape[0]][prompt_mask] = float(args.alpha_prompt)
    gen_weights = weights[prefix_ids.shape[0] :]
    gen_weights[generated_is_mask] = alpha_mask
    gen_weights[generated_is_special] = alpha_special
    gen_weights[generated_is_normal] = alpha_normal
    return weights


def compute_confidence(logits, x0, remasking):
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise NotImplementedError(remasking)


def save_reweighted_plots(result, output_dir, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "attention_distribution": output_dir / "attention_distribution.png",
        "step_attention_distribution": output_dir / "step_attention_distribution.png",
        "layer_step_visual_attention_heatmap": output_dir / "layer_step_visual_attention_heatmap.png",
        "layer_step_prompt_text_attention_heatmap": output_dir / "layer_step_prompt_text_attention_heatmap.png",
        "layer_step_generated_text_attention_heatmap": output_dir / "layer_step_generated_text_attention_heatmap.png",
        "gen_text_attention_distribution": output_dir / "gen_text_attention_distribution.png",
        "visual_attention_over_time": output_dir / "visual_attention_over_time.png",
        "visual_attention_over_steps": output_dir / "visual_attention_over_steps.png",
    }
    save_attention_distribution_plot(result["token_stats"], paths["attention_distribution"], dpi)
    save_step_attention_distribution_plot(result["step_stats"], paths["step_attention_distribution"], dpi)
    save_layer_step_heatmap(result["layer_step_stats"], "visual", paths["layer_step_visual_attention_heatmap"], dpi)
    save_layer_step_heatmap(result["layer_step_stats"], "prompt_text", paths["layer_step_prompt_text_attention_heatmap"], dpi)
    save_layer_step_heatmap(result["layer_step_stats"], "generated_text", paths["layer_step_generated_text_attention_heatmap"], dpi)
    save_gen_text_attention_distribution_plot(result["gen_text_step_stats"], paths["gen_text_attention_distribution"], dpi)
    save_visual_attention_curve(result["token_stats"], paths["visual_attention_over_time"], dpi)
    save_visual_attention_over_steps(result["step_stats"], paths["visual_attention_over_steps"], dpi)
    return {name: str(path) for name, path in paths.items()}


def analyze_single_example_reweighted(model, tokenizer, image_processor, args, output_dir):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    prompt = build_prompt(args.question, args.conv_template)
    prefix_embeds, prefix_input_ids, visual_mask = prepare_multimodal_prefix(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_path=args.image,
        prompt_text=prompt,
        device=device,
        dtype=dtype,
    )

    core_model = model.get_model()
    layers_to_capture = list(range(len(core_model.transformer.blocks)))
    special_token_ids = get_special_token_ids(tokenizer)

    with torch.no_grad():
        past_key_values = core_model(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
        x = torch.full((1, args.max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)

        if args.max_new_tokens % args.block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length.")

        num_blocks = args.max_new_tokens // args.block_length
        steps = args.max_new_tokens // num_blocks
        if args.step_ratio is not None:
            steps = int(steps * args.step_ratio)
        if steps <= 0:
            raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

        token_stats = {}
        step_stats = []
        layer_step_stats = []
        gen_text_step_stats = []
        category_weight_state = {}

        with patch_category_reweight_attention(model, category_weight_state):
            for block_idx in range(num_blocks):
                block_slice = slice(block_idx * args.block_length, (block_idx + 1) * args.block_length)
                block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
                schedule_kwargs = {"shift": args.schedule_shift} if args.schedule == "shift" else None
                num_transfer_tokens = get_num_transfer_tokens_sch(
                    block_mask_index,
                    steps,
                    schedule=args.schedule,
                    schedule_kwargs=schedule_kwargs,
                )

                for step_idx in range(num_transfer_tokens.shape[1]):
                    mask_index = x == MASK_TOKEN_ID
                    block_mask_index = mask_index[:, block_slice]
                    if block_mask_index.sum().item() == 0:
                        continue

                    category_weight_state["weights"] = build_reweight_weights(
                        prefix_input_ids,
                        visual_mask,
                        x[0],
                        special_token_ids,
                        args,
                    )
                    category_weight_state["query_is_mask"] = mask_index[0]
                    category_weight_state["attention_store"] = {}

                    current_embeds = core_model.transformer.wte(x)
                    logits = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits
                    logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    x0_p = compute_confidence(logits, x0, args.remasking)
                    x0_p[:, (block_idx + 1) * args.block_length :] = -torch.inf
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -torch.inf)

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for batch_idx in range(confidence.shape[0]):
                        k = int(num_transfer_tokens[batch_idx, step_idx].item())
                        _, select_index = torch.topk(confidence[batch_idx], k=k)
                        transfer_index[batch_idx, select_index] = True

                    selected_queries = torch.where(transfer_index[0])[0]
                    collector = QueryAttentionCollector(
                        prefix_ids=prefix_input_ids,
                        visual_mask=visual_mask,
                        gen_tokens=x[0],
                        special_token_ids=special_token_ids,
                        selected_queries=selected_queries,
                    )

                    current_layer_step_stats = []
                    attention_store = category_weight_state["attention_store"]
                    for layer_idx in layers_to_capture:
                        layer_key = f"layer_{layer_idx}"
                        if layer_key in attention_store and attention_store[layer_key]:
                            layer_attn = attention_store[layer_key][0]
                            collector.record(layer_attn)
                            layer_collector = QueryAttentionCollector(
                                prefix_ids=prefix_input_ids,
                                visual_mask=visual_mask,
                                gen_tokens=x[0],
                                special_token_ids=special_token_ids,
                                selected_queries=selected_queries,
                            )
                            layer_collector.record(layer_attn)
                            current_layer_step_stats.append(average_query_stats(layer_collector.finalize()))
                        else:
                            current_layer_step_stats.append({name: 0.0 for name in CATEGORY_ORDER})
                    layer_step_stats.append(current_layer_step_stats)

                    finalized = collector.finalize()
                    finalized_gen_text = collector.finalize_gen_text()
                    for query_pos, stats in finalized.items():
                        token_stats[query_pos] = stats
                    if finalized:
                        step_stats.append({name: sum(v[name] for v in finalized.values()) / len(finalized) for name in CATEGORY_ORDER})
                    if finalized_gen_text:
                        gen_text_step_stats.append(
                            {name: sum(v[name] for v in finalized_gen_text.values()) / len(finalized_gen_text) for name in GEN_TEXT_CATEGORY_ORDER}
                        )

                    x[transfer_index] = x0[transfer_index]

        missing_positions = [idx for idx in range(args.max_new_tokens) if idx not in token_stats]
        if missing_positions:
            raise RuntimeError(f"Some generated token positions were never recorded: {missing_positions[:10]}")

        ordered_token_stats = [token_stats[idx] for idx in range(args.max_new_tokens)]
        final_text = tokenizer.batch_decode(x, skip_special_tokens=False)[0].replace("<|endoftext|>", "")

    result = {
        "image_path": args.image,
        "question": args.question,
        "final_text": final_text,
        "token_stats": ordered_token_stats,
        "step_stats": step_stats,
        "layer_step_stats": layer_step_stats,
        "gen_text_step_stats": gen_text_step_stats,
    }
    result.update(save_reweighted_plots(result, output_dir, args.dpi))
    return result


def save_visual_step_comparison(baseline_result, reweighted_result, output_path, dpi):
    baseline = baseline_result["step_stats"]
    reweighted = reweighted_result["step_stats"]
    n = min(len(baseline), len(reweighted))
    x = list(range(n))
    plt.figure(figsize=(14, 4.8))
    plt.plot(x, [baseline[idx]["visual"] for idx in x], label="Baseline", linewidth=1.8)
    plt.plot(x, [reweighted[idx]["visual"] for idx in x], label="Reweighted", linewidth=1.8)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Visual Attention")
    plt.title("Visual Attention Before vs After Reweighting")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_step_delta_comparison(baseline_result, reweighted_result, output_path, dpi):
    baseline = baseline_result["step_stats"]
    reweighted = reweighted_result["step_stats"]
    n = min(len(baseline), len(reweighted))
    x = list(range(n))
    plt.figure(figsize=(14, 4.8))
    for name in CATEGORY_ORDER:
        delta = [reweighted[idx][name] - baseline[idx][name] for idx in x]
        plt.plot(x, delta, label=f"{name} delta", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Reweighted - Baseline Attention")
    plt.title("Attention Distribution Delta Over Steps")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def summarize_step_delta(baseline_result, reweighted_result):
    baseline = baseline_result["step_stats"]
    reweighted = reweighted_result["step_stats"]
    n = min(len(baseline), len(reweighted))
    if n == 0:
        return {name: {"mean_delta": 0.0, "max_abs_delta": 0.0} for name in CATEGORY_ORDER}

    summary = {}
    for name in CATEGORY_ORDER:
        deltas = [reweighted[idx][name] - baseline[idx][name] for idx in range(n)]
        summary[name] = {
            "mean_delta": sum(deltas) / n,
            "max_abs_delta": max(abs(value) for value in deltas),
            "first_delta": deltas[0],
            "last_delta": deltas[-1],
        }
    return summary


def run_compare(args):
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "baseline"
    reweighted_dir = output_dir / "reweighted"
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        args.model_name,
        device_map=f"{args.device}:0" if args.device.startswith("cuda") else args.device,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))

    baseline_result = analyze_single_example(
        model,
        tokenizer,
        image_processor,
        args,
        args.image,
        args.question,
        baseline_dir,
        save_plots=True,
    )
    reweighted_result = analyze_single_example_reweighted(model, tokenizer, image_processor, args, reweighted_dir)

    comparison_path = output_dir / "compare_visual_attention_over_steps.png"
    save_visual_step_comparison(baseline_result, reweighted_result, comparison_path, args.dpi)
    delta_path = output_dir / "compare_attention_delta_over_steps.png"
    save_step_delta_comparison(baseline_result, reweighted_result, delta_path, args.dpi)
    step_delta_summary = summarize_step_delta(baseline_result, reweighted_result)

    summary = {
        "alpha_prompt": float(args.alpha_prompt),
        "alpha_visual": float(args.alpha_visual),
        "alpha_generated": float(args.alpha_generated),
        "alpha_mask": float(args.alpha_generated if args.alpha_mask is None else args.alpha_mask),
        "alpha_normal": float(args.alpha_generated if args.alpha_normal is None else args.alpha_normal),
        "alpha_special": float(args.alpha_generated if args.alpha_special is None else args.alpha_special),
        "baseline": baseline_result,
        "reweighted": reweighted_result,
        "compare_visual_attention_over_steps": str(comparison_path),
        "compare_attention_delta_over_steps": str(delta_path),
        "step_delta_summary": step_delta_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved baseline plots under: {baseline_dir}")
    print(f"Saved reweighted plots under: {reweighted_dir}")
    print(f"Saved comparison plot: {comparison_path}")
    print(f"Saved delta plot: {delta_path}")
    print("Step delta summary:")
    print(json.dumps(step_delta_summary, ensure_ascii=False, indent=2))
    print("Baseline final text:")
    print(baseline_result["final_text"])
    print("Reweighted final text:")
    print(reweighted_result["final_text"])


if __name__ == "__main__":
    run_compare(parse_args())
