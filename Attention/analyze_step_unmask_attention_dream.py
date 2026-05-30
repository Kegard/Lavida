import argparse
import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


MASK_TOKEN_ID = 151666
CATEGORY_ORDER = ["visual", "prompt_text", "generated_text"]
CATEGORY_LABELS = {
    "visual": "Visual",
    "prompt_text": "Prompt Text",
    "generated_text": "Generated Text",
}
CATEGORY_COLORS = {
    "visual": "#ff8c8c",
    "prompt_text": "#9be7e8",
    "generated_text": "#7ae0d6",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze attention-source proportions for each generated token at the moment "
            "it is newly unmasked during Dream discrete diffusion generation."
        )
    )
    parser.add_argument("--pretrained", default="weight/dream")
    parser.add_argument("--model-name", default="llava_dream")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default="images/dog.png")
    parser.add_argument("--question", default="Describe the image in detail.")
    parser.add_argument("--conv-template", default="dream")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--schedule", default="none")
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--output-dir", default="Attention/outputs/step_unmask_attention_dream")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--dataset-path", default=None, help="Optional dataset path for direct loading via datasets.load_dataset, e.g. lmms-lab/LMMs-Eval-Lite.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset config name, e.g. textvqa_val.")
    parser.add_argument("--split", default="validation", help="Dataset split name.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of dataset samples to process.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip the first N dataset samples before visualization.")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_dataset_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def construct_textvqa_question(doc):
    return f"{doc['question'].capitalize()}\nAnswer the question using a single word or phrase."


def extract_textvqa_images(doc):
    return [doc["image"].convert("RGB")]


def build_prompt(question: str, conv_template: str) -> str:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def compute_visual_token_info(model):
    vision_tower = model.get_vision_tower()
    if hasattr(vision_tower, "num_patches_per_side"):
        patches_per_side = int(vision_tower.num_patches_per_side)
    elif hasattr(vision_tower, "num_patches"):
        patches_per_side = int(math.sqrt(int(vision_tower.num_patches)))
    else:
        raise ValueError("Cannot infer visual token grid from vision tower.")
    return {
        "height": patches_per_side,
        "width": patches_per_side,
        "total_tokens": patches_per_side * patches_per_side,
    }


def resolve_visual_positions(image_positions, vis_info):
    expected = int(vis_info["total_tokens"])
    if expected > 0 and expected <= len(image_positions):
        return image_positions[:expected]
    return image_positions


def prepare_multimodal_prefix_from_image(model, tokenizer, image_processor, image, prompt_text, device, dtype):
    image = image.convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=dtype, device=device) for _image in image_tensor]

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    ret = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
        modalities=["image"],
        image_sizes=[image.size],
        return_inputs=True,
    )
    prefix_embeds, prefix_input_ids = ret[4], ret[-1]

    valid_prefix = prefix_input_ids[0] != IGNORE_INDEX
    prefix_embeds = prefix_embeds[:, valid_prefix, :]
    prefix_input_ids = prefix_input_ids[0, valid_prefix]

    image_positions_raw = torch.where(prefix_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    vis_info = compute_visual_token_info(model)
    visual_positions = resolve_visual_positions(image_positions_raw, vis_info)
    visual_mask = torch.zeros_like(prefix_input_ids, dtype=torch.bool)
    if visual_positions:
        visual_mask[torch.tensor(visual_positions, device=prefix_input_ids.device, dtype=torch.long)] = True

    return prefix_embeds, prefix_input_ids, visual_mask


class QueryAttentionCollector:
    def __init__(self, prefix_ids: torch.Tensor, visual_mask: torch.Tensor, gen_length: int, selected_queries: torch.Tensor):
        self.selected_queries = selected_queries.detach().cpu().to(dtype=torch.long)
        self.query_positions = [int(pos) for pos in self.selected_queries.tolist()]

        prefix_ids = prefix_ids.detach().cpu().to(dtype=torch.long)
        prefix_is_visual = visual_mask.detach().cpu().to(dtype=torch.bool)
        prefix_is_prompt_text = (~prefix_is_visual) & (prefix_ids != IGNORE_INDEX)
        generated_region = torch.ones(gen_length, dtype=torch.bool)

        self.key_masks = {
            "visual": prefix_is_visual,
            "prompt_text": prefix_is_prompt_text,
            "generated_text": generated_region,
        }
        self.query_category_sums = {
            query_pos: {name: 0.0 for name in CATEGORY_ORDER}
            for query_pos in self.query_positions
        }

    def record(self, attn_probs: torch.Tensor):
        if self.selected_queries.numel() == 0:
            return

        probs = attn_probs[0, :, self.selected_queries, :].to(torch.float32).detach().cpu()
        prefix_len = self.key_masks["visual"].shape[0]

        for local_idx, query_pos in enumerate(self.query_positions):
            query_probs = probs[:, local_idx, :]
            for name in CATEGORY_ORDER:
                prefix_mask = self.key_masks[name]
                if name == "generated_text":
                    expanded_mask = torch.cat(
                        [
                            torch.zeros(prefix_len, dtype=torch.bool),
                            prefix_mask,
                        ]
                    )
                else:
                    expanded_mask = torch.cat(
                        [
                            prefix_mask,
                            torch.zeros(query_probs.shape[-1] - prefix_len, dtype=torch.bool),
                        ]
                    )
                self.query_category_sums[query_pos][name] += query_probs[:, expanded_mask].sum().item()

    def finalize(self):
        result = {}
        for query_pos, sums in self.query_category_sums.items():
            total = sum(sums.values())
            if total <= 0:
                result[query_pos] = {name: 0.0 for name in CATEGORY_ORDER}
            else:
                result[query_pos] = {name: sums[name] / total for name in CATEGORY_ORDER}
        return result


def save_attention_distribution_plot(token_stats, output_path: Path, dpi: int):
    x = list(range(len(token_stats)))
    bottoms = [0.0] * len(token_stats)

    plt.figure(figsize=(14, 4.8))
    for name in CATEGORY_ORDER:
        values = [token_stats[idx][name] for idx in x]
        plt.bar(
            x,
            values,
            bottom=bottoms,
            width=0.92,
            color=CATEGORY_COLORS[name],
            edgecolor="none",
            label=CATEGORY_LABELS[name],
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    plt.ylim(0.0, 1.0)
    plt.xlim(-0.5, len(token_stats) - 0.5)
    plt.xlabel("Generated Token Index")
    plt.ylabel("Attention Weight")
    plt.title("Attention Distribution")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_step_attention_distribution_plot(step_stats, output_path: Path, dpi: int):
    x = list(range(len(step_stats)))
    bottoms = [0.0] * len(step_stats)

    plt.figure(figsize=(14, 4.8))
    for name in CATEGORY_ORDER:
        values = [step_stats[idx][name] for idx in x]
        plt.bar(
            x,
            values,
            bottom=bottoms,
            width=0.92,
            color=CATEGORY_COLORS[name],
            edgecolor="none",
            label=CATEGORY_LABELS[name],
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    plt.ylim(0.0, 1.0)
    plt.xlim(-0.5, len(step_stats) - 0.5)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Attention Weight")
    plt.title("Attention Distribution Over Steps")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_visual_attention_curve(token_stats, output_path: Path, dpi: int):
    x = list(range(len(token_stats)))
    y = [token_stats[idx]["visual"] for idx in x]

    plt.figure(figsize=(14, 4.8))
    plt.plot(x, y, color="purple", linewidth=1.7, marker="o", markersize=2.8)
    plt.xlabel("Generated Token Index")
    plt.ylabel("Visual Attention")
    plt.title("Visual Attention Over Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_visual_attention_over_steps(step_stats, output_path: Path, dpi: int):
    x = list(range(len(step_stats)))
    y = [step_stats[idx]["visual"] for idx in x]

    plt.figure(figsize=(14, 4.8))
    plt.plot(x, y, color="purple", linewidth=1.7, marker="o", markersize=2.8)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Visual Attention")
    plt.title("Visual Attention Over Steps")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def average_stat_sequences(stat_sequences):
    if not stat_sequences:
        raise ValueError("No statistic sequences to average.")
    min_len = min(len(seq) for seq in stat_sequences)
    if min_len <= 0:
        raise ValueError("Statistic sequences are empty.")

    averaged = []
    for idx in range(min_len):
        averaged_item = {}
        for name in CATEGORY_ORDER:
            averaged_item[name] = sum(seq[idx][name] for seq in stat_sequences) / len(stat_sequences)
        averaged.append(averaged_item)
    return averaged


def build_current_state_input(model, prefix_embeds, state_tokens):
    state_embeds = model.get_model().embed_tokens(state_tokens)
    return torch.cat([prefix_embeds, state_embeds], dim=1)


def collect_step_attention(model, full_inputs_embeds, selected_queries, prefix_input_ids, visual_mask, gen_length):
    with torch.no_grad():
        outputs = model.forward_dream(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=full_inputs_embeds,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    collector = QueryAttentionCollector(
        prefix_ids=prefix_input_ids,
        visual_mask=visual_mask,
        gen_length=gen_length,
        selected_queries=selected_queries,
    )
    prefix_len = prefix_input_ids.shape[0]
    query_positions = selected_queries + prefix_len

    for layer_attn in outputs.attentions:
        collector.record(layer_attn[:, :, query_positions, :])
    return collector.finalize()


def analyze_single_example(model, tokenizer, image_processor, args, image, question: str):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    prompt = build_prompt(question, args.conv_template)
    prefix_embeds, prefix_input_ids, visual_mask = prepare_multimodal_prefix_from_image(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image=image,
        prompt_text=prompt,
        device=device,
        dtype=dtype,
    )

    image_tensor = process_images([image.convert("RGB")], image_processor, model.config)
    image_tensor = [_image.to(dtype=dtype, device=device) for _image in image_tensor]
    image_sizes = [image.size]

    with torch.no_grad():
        outputs = model.generate(
            None,
            images=image_tensor,
            image_sizes=image_sizes,
            max_new_tokens=args.max_new_tokens,
            steps=args.max_new_tokens,
            step_ratio=args.step_ratio,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.alg,
            alg_temp=args.alg_temp,
            prefix_lm=True,
            output_history=True,
            schedule=None if args.schedule == "none" else args.schedule,
            schedule_kwargs={"shift": args.schedule_shift} if args.schedule == "shift" else None,
        )

    sequences = outputs.sequences
    history = outputs.history or []
    final_tokens = sequences[0]
    gen_length = final_tokens.shape[0]

    previous_state = torch.full((gen_length,), MASK_TOKEN_ID, dtype=torch.long, device=final_tokens.device)
    if history:
        previous_state[:1] = history[0][0, :1]

    token_stats = {}
    step_stats = []

    for step_idx, step_state_batch in enumerate(history):
        step_state = step_state_batch[0]
        new_mask = (previous_state == MASK_TOKEN_ID) & (step_state != MASK_TOKEN_ID)
        selected_queries = torch.where(new_mask)[0]
        if selected_queries.numel() == 0:
            previous_state = step_state.clone()
            continue

        current_input_embeds = build_current_state_input(model, prefix_embeds, previous_state.unsqueeze(0))
        current_step_stats = collect_step_attention(
            model=model,
            full_inputs_embeds=current_input_embeds,
            selected_queries=selected_queries,
            prefix_input_ids=prefix_input_ids,
            visual_mask=visual_mask,
            gen_length=gen_length,
        )

        for query_pos, stats in current_step_stats.items():
            token_stats[query_pos] = stats

        step_mean = {}
        for name in CATEGORY_ORDER:
            step_mean[name] = sum(v[name] for v in current_step_stats.values()) / len(current_step_stats)
        step_stats.append(step_mean)
        previous_state = step_state.clone()

    missing_positions = [idx for idx in range(gen_length) if idx not in token_stats]
    if missing_positions:
        raise RuntimeError(f"Some generated token positions were never recorded: {missing_positions[:10]}")

    ordered_token_stats = [token_stats[idx] for idx in range(gen_length)]
    final_text = tokenizer.batch_decode(sequences, skip_special_tokens=False)[0].replace("<|endoftext|>", "")
    return {
        "question": question,
        "final_text": final_text,
        "token_stats": ordered_token_stats,
        "step_stats": step_stats,
    }


def run_generation_analysis(args):
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
        device_map=args.device_map,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(dtype)

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
            result = analyze_single_example(model, tokenizer, image_processor, args, image, question)
            batch_token_stats.append(result["token_stats"])
            batch_step_stats.append(result["step_stats"])
            print(f"Processed sample {sample_idx + args.sample_offset}")
            print(result["final_text"])

        avg_token_stats = average_stat_sequences(batch_token_stats)
        avg_step_stats = average_stat_sequences(batch_step_stats)
    else:
        image = Image.open(args.image).convert("RGB")
        result = analyze_single_example(model, tokenizer, image_processor, args, image, args.question)
        avg_token_stats = result["token_stats"]
        avg_step_stats = result["step_stats"]
        print("Final generation:")
        print(result["final_text"])

    attention_dist_path = output_dir / "attention_distribution.png"
    step_attention_dist_path = output_dir / "step_attention_distribution.png"
    visual_curve_path = output_dir / "visual_attention_over_time.png"
    visual_step_curve_path = output_dir / "visual_attention_over_steps.png"
    save_attention_distribution_plot(avg_token_stats, attention_dist_path, args.dpi)
    save_step_attention_distribution_plot(avg_step_stats, step_attention_dist_path, args.dpi)
    save_visual_attention_curve(avg_token_stats, visual_curve_path, args.dpi)
    save_visual_attention_over_steps(avg_step_stats, visual_step_curve_path, args.dpi)

    print(f"Saved: {attention_dist_path}")
    print(f"Saved: {step_attention_dist_path}")
    print(f"Saved: {visual_curve_path}")
    print(f"Saved: {visual_step_curve_path}")


if __name__ == "__main__":
    run_generation_analysis(parse_args())
