import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scale_Attention.taca_patch import (
    build_taca_scales_from_multimodal_inputs,
    parse_bool_like,
    patch_taca_attention,
)

from Attention.analyze_step_unmask_attention import (
    average_stat_sequences,
    analyze_single_dataset_example,
    analyze_single_example,
    build_prompt,
    construct_textvqa_question,
    extract_textvqa_images,
    get_torch_dtype,
    load_dataset_split,
    load_pretrained_model,
    save_attention_distribution_plot,
    save_step_attention_distribution_plot,
    save_visual_attention_curve,
    save_visual_attention_over_steps,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze step-wise attention distributions under TACA-style decode-time "
            "logit scaling for LaViDa."
        )
    )
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
    parser.add_argument("--output-dir", default="Scale_Attention/outputs/step_unmask_attention_taca")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--taca-enable", default=True)
    parser.add_argument("--taca-gamma-prompt", type=float, default=1.5)
    parser.add_argument("--taca-gamma-visual", type=float, default=1.5)
    parser.add_argument("--taca-scale-generated", type=float, default=1.0)
    return parser.parse_args()


def analyze_single_example_taca(model, tokenizer, image_processor, args, image_path: str, question: str, output_dir: Path = None, save_plots: bool = True):
    prompt = build_prompt(question, args.conv_template)
    input_ids = tokenizer.image_token if False else None  # placeholder to keep lint calm
    _ = input_ids
    # Reuse the original analysis implementation, but run generation under the TACA patch.
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(dtype=get_torch_dtype(args.torch_dtype), device=args.device)
    input_ids = __import__("llava.mm_utils", fromlist=["tokenizer_image_token"]).tokenizer_image_token(
        prompt, tokenizer, __import__("llava.constants", fromlist=["IMAGE_TOKEN_INDEX"]).IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)
    scales, _meta = build_taca_scales_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        gen_len=int(args.max_new_tokens),
        gamma_prompt=float(args.taca_gamma_prompt),
        gamma_visual=float(args.taca_gamma_visual),
        scale_generated=float(args.taca_scale_generated),
    )
    with patch_taca_attention(model, scales):
        return analyze_single_example(model, tokenizer, image_processor, args, image_path, question, output_dir, save_plots)


def analyze_single_dataset_example_taca(model, tokenizer, image_processor, args, image, question: str):
    prompt = build_prompt(question, args.conv_template)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(dtype=get_torch_dtype(args.torch_dtype), device=args.device)
    input_ids = __import__("llava.mm_utils", fromlist=["tokenizer_image_token"]).tokenizer_image_token(
        prompt, tokenizer, __import__("llava.constants", fromlist=["IMAGE_TOKEN_INDEX"]).IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)
    scales, _meta = build_taca_scales_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        gen_len=int(args.max_new_tokens),
        gamma_prompt=float(args.taca_gamma_prompt),
        gamma_visual=float(args.taca_gamma_visual),
        scale_generated=float(args.taca_scale_generated),
    )
    with patch_taca_attention(model, scales):
        return analyze_single_dataset_example(model, tokenizer, image_processor, args, image, question)


def run_generation_analysis_taca(args):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    output_dir = Path(args.output_dir)
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
        device_map=f"{device}:0" if device.startswith("cuda") else device,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(dtype)

    if parse_bool_like(args.taca_enable) is False:
        raise ValueError("This script is intended for TACA-enabled analysis. Use the original analysis script for baseline.")

    if args.dataset_path is not None:
        dataset = load_dataset_split(args.dataset_path, args.dataset_name, args.split)
        if args.sample_offset > 0:
            dataset = dataset.select(range(args.sample_offset, len(dataset)))
        if args.max_samples > 0:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        if len(dataset) == 0:
            raise ValueError("No dataset samples selected for processing.")

        print(f"Processing {len(dataset)} dataset samples from {args.dataset_path}")
        batch_token_stats = []
        batch_step_stats = []
        for sample_idx, sample in enumerate(dataset):
            question = construct_textvqa_question(sample)
            image = extract_textvqa_images(sample)[0]
            result = analyze_single_dataset_example_taca(model, tokenizer, image_processor, args, image, question)
            batch_token_stats.append(result["token_stats"])
            batch_step_stats.append(result["step_stats"])
            print(f"Processed sample {sample_idx + args.sample_offset}")
            print(result["final_text"])

        avg_token_stats = average_stat_sequences(batch_token_stats)
        avg_step_stats = average_stat_sequences(batch_step_stats)

        attention_dist_path = output_dir / "attention_distribution.png"
        step_attention_dist_path = output_dir / "step_attention_distribution.png"
        visual_curve_path = output_dir / "visual_attention_over_time.png"
        visual_step_curve_path = output_dir / "visual_attention_over_steps.png"
        save_attention_distribution_plot(avg_token_stats, attention_dist_path, args.dpi)
        save_step_attention_distribution_plot(avg_step_stats, step_attention_dist_path, args.dpi)
        save_visual_attention_curve(avg_token_stats, visual_curve_path, args.dpi)
        save_visual_attention_over_steps(avg_step_stats, visual_step_curve_path, args.dpi)

        summary = {
            "taca_gamma_prompt": float(args.taca_gamma_prompt),
            "taca_gamma_visual": float(args.taca_gamma_visual),
            "taca_scale_generated": float(args.taca_scale_generated),
            "num_samples": len(dataset),
            "attention_distribution": str(attention_dist_path),
            "step_attention_distribution": str(step_attention_dist_path),
            "visual_attention_over_time": str(visual_curve_path),
            "visual_attention_over_steps": str(visual_step_curve_path),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {attention_dist_path}")
        print(f"Saved: {step_attention_dist_path}")
        print(f"Saved: {visual_curve_path}")
        print(f"Saved: {visual_step_curve_path}")
        return

    result = analyze_single_example_taca(model, tokenizer, image_processor, args, args.image, args.question, output_dir)
    summary = {
        "taca_gamma_prompt": float(args.taca_gamma_prompt),
        "taca_gamma_visual": float(args.taca_gamma_visual),
        "taca_scale_generated": float(args.taca_scale_generated),
        "attention_distribution": result["attention_distribution"],
        "step_attention_distribution": result["step_attention_distribution"],
        "visual_attention_over_time": result["visual_attention_over_time"],
        "visual_attention_over_steps": result["visual_attention_over_steps"],
        "final_text": result["final_text"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {result['attention_distribution']}")
    print(f"Saved: {result['step_attention_distribution']}")
    print(f"Saved: {result['visual_attention_over_time']}")
    print(f"Saved: {result['visual_attention_over_steps']}")
    print("Final generation:")
    print(result["final_text"])


if __name__ == "__main__":
    args = parse_args()
    run_generation_analysis_taca(args)
