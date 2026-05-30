import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scale_Attention.layer_selection import (
    collect_layer_selection_metrics,
    select_recovery_and_emergence_layers,
    strip_visual_maps,
)


MASK_TOKEN_ID = 126336
LETTER_MAP = "ABCDEFG"


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_choice_block(choices):
    return "\n".join(f"({LETTER_MAP[i]}) {choice}" for i, choice in enumerate(choices))


def build_m3cot_prompt(doc, prompt_style: str) -> str:
    parts = []
    context = (doc.get("context") or "").strip()
    if context:
        parts.append(f"[Context]\n{context}")
    parts.append(f"[Question]\n{doc['question']}")
    parts.append(f"[Choices]\n{build_choice_block(doc['choices'])}")
    base = "\n".join(parts)

    if prompt_style == "direct":
        return base + "\n\nAnswer with the option's letter from the given choices directly."
    if prompt_style == "cot":
        return base + "\n\nThink step by step and then provide your final answer in the format [Answer] (X)."
    if prompt_style == "dsp":
        return (
            base
            + "\n\nFirst describe the image information relevant to the question. "
            + "Then reason briefly and provide the final answer in the format [Answer] (X)."
        )
    if prompt_style == "ccot":
        return (
            base
            + "\n\nFirst identify the relevant objects, attributes, and relationships in the image as a compact scene graph. "
            + "Then solve the question and provide the final answer in the format [Answer] (X)."
        )
    raise ValueError(f"Unsupported M3CoT prompt style: {prompt_style}")


def sanitize_filename(value) -> str:
    text = str(value)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "sample"


def sanitize_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(value) for value in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and visualize LaSCD-style recovery/emergence layers from visual-attention structure."
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
    parser.add_argument("--min-visual-mass", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="Scale_Attention/outputs/lascd_layer_selection")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--m3cot", action="store_true", help="Analyze a sampled subset from M3CoT instead of a single image/question.")
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--save-sample-plots", action="store_true", help="Also save full per-sample plots for each M3CoT item.")
    parser.add_argument("--trace-answer-logits", action="store_true", help="Track answer-option logits for every layer on M3CoT samples.")
    return parser.parse_args()


def compute_confidence(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
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


def save_heatmap(matrix, output_path: Path, title: str, colorbar_label: str, dpi: int, cmap: str = "viridis"):
    data = torch.tensor(matrix, dtype=torch.float32)
    if data.ndim != 2 or data.numel() == 0:
        raise ValueError(f"Expected a non-empty 2D matrix for {title}.")

    width = max(8.0, min(28.0, 0.28 * data.shape[1] + 4.0))
    height = max(5.0, min(18.0, 0.22 * data.shape[0] + 2.5))
    plt.figure(figsize=(width, height))
    im = plt.imshow(data.numpy(), aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_selected_layer_curve(step_records, output_path: Path, dpi: int):
    x = list(range(len(step_records)))
    recovery = [record["selection"]["recovery_layer"] for record in step_records]
    emergence = [record["selection"]["emergence_layer"] for record in step_records]

    plt.figure(figsize=(14, 4.8))
    plt.plot(x, recovery, label="Recovery Layer", linewidth=1.8, marker="o", markersize=3)
    plt.plot(x, emergence, label="Emergence Layer", linewidth=1.8, marker="s", markersize=3)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Layer Index")
    plt.title("Selected Recovery and Emergence Layers Over Steps")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_layer_score_curve(step_records, output_path: Path, dpi: int):
    x = list(range(len(step_records)))
    recovery = [record["selection"]["recovery_score"] for record in step_records]
    emergence = [record["selection"]["emergence_score"] for record in step_records]

    plt.figure(figsize=(14, 4.8))
    plt.plot(x, recovery, label="Recovery Score", linewidth=1.8)
    plt.plot(x, emergence, label="Emergence Score", linewidth=1.8)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Visual Mass x Laplacian Energy")
    plt.title("Selected Layer Scores Over Steps")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_answer_margin_heatmap(step_records, output_path: Path, dpi: int):
    matrix = []
    for record in step_records:
        trace = record.get("layer_answer_trace")
        if not trace:
            continue
        matrix.append([float(item["correct_vs_max_wrong_margin"]) for item in trace])
    if not matrix:
        return
    save_heatmap(
        torch.tensor(matrix, dtype=torch.float32).transpose(0, 1),
        output_path,
        "Correct Answer Margin by Layer and Step",
        "Correct Logit - Max Wrong Logit",
        dpi,
        cmap="coolwarm",
    )


def save_answer_margin_curve(step_records, output_path: Path, dpi: int):
    traces = [record.get("layer_answer_trace") for record in step_records if record.get("layer_answer_trace")]
    if not traces:
        return
    num_layers = min(len(trace) for trace in traces)
    means = []
    for layer_idx in range(num_layers):
        values = [float(trace[layer_idx]["correct_vs_max_wrong_margin"]) for trace in traces]
        means.append(sum(values) / len(values))

    plt.figure(figsize=(12, 4.8))
    plt.plot(list(range(num_layers)), means, linewidth=1.8, marker="o", markersize=3)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Layer Index")
    plt.ylabel("Correct Logit - Max Wrong Logit")
    plt.title("Average Correct Answer Margin Across Steps")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_visual_map(visual_map: torch.Tensor, output_path: Path, title: str, dpi: int):
    data = visual_map.detach().cpu().to(dtype=torch.float32)
    total = data.sum()
    if float(total.item()) > 0.0:
        data = data / total.clamp_min(1e-12)

    plt.figure(figsize=(5.2, 4.8))
    im = plt.imshow(data.numpy(), cmap="magma")
    plt.colorbar(im, label="Normalized Visual Attention")
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def find_metric(layer_metrics, layer_idx: int):
    for item in layer_metrics:
        if int(item["layer"]) == int(layer_idx):
            return item
    raise KeyError(f"Layer {layer_idx} not found in metrics.")


def get_option_token_ids(tokenizer, num_choices: int) -> List[int]:
    token_ids = []
    for idx in range(num_choices):
        letter = LETTER_MAP[idx]
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            encoded = tokenizer.encode(" " + letter, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Could not tokenize option letter {letter!r}.")
        token_ids.append(int(encoded[-1]))
    return token_ids


def answer_to_index(answer, choices) -> int:
    if isinstance(answer, int):
        return int(answer)
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in LETTER_MAP:
        return LETTER_MAP.index(text.upper())
    if text in choices:
        return list(choices).index(text)
    raise ValueError(f"Cannot map answer {answer!r} to a choice index.")


def layer_logits_from_hidden(core_model, hidden_state: torch.Tensor, option_token_ids: List[int], query_positions: torch.Tensor):
    state = core_model.transformer.ln_f(hidden_state)
    if core_model.config.weight_tying:
        option_weight = core_model.transformer.wte.weight[option_token_ids].to(device=state.device, dtype=state.dtype)
        logits = torch.matmul(state, option_weight.transpose(0, 1))
    else:
        full_logits = core_model.transformer.ff_out(state)
        logits = full_logits[..., option_token_ids]
    if core_model.config.scale_logits:
        logits = logits * (1.0 / (core_model.config.d_model ** 0.5))

    query_positions = query_positions.to(device=logits.device, dtype=torch.long).view(-1)
    if query_positions.numel() == 0:
        return torch.zeros((len(option_token_ids),), dtype=torch.float32)
    query_positions = query_positions[(query_positions >= 0) & (query_positions < logits.shape[1])]
    if query_positions.numel() == 0:
        return torch.zeros((len(option_token_ids),), dtype=torch.float32)
    return logits[0, query_positions, :].mean(dim=0).detach().cpu().to(dtype=torch.float32)


def compute_layer_answer_trace(
    core_model,
    hidden_states,
    selected_queries: torch.Tensor,
    option_token_ids: List[int],
    correct_choice_idx: int,
    predicted_choice_idx: int,
):
    num_layers = len(core_model.transformer.blocks)
    records = []
    for layer_idx, hidden_state in enumerate(hidden_states[1 : num_layers + 1]):
        option_logits = layer_logits_from_hidden(
            core_model,
            hidden_state,
            option_token_ids=option_token_ids,
            query_positions=selected_queries,
        )
        other_indices = [idx for idx in range(len(option_token_ids)) if idx != int(correct_choice_idx)]
        max_wrong_idx = max(other_indices, key=lambda idx: float(option_logits[idx])) if other_indices else int(correct_choice_idx)
        correct_logit = float(option_logits[int(correct_choice_idx)].item())
        predicted_logit = float(option_logits[int(predicted_choice_idx)].item())
        max_wrong_logit = float(option_logits[int(max_wrong_idx)].item())
        records.append(
            {
                "layer": int(layer_idx),
                "option_logits": [float(value) for value in option_logits.tolist()],
                "correct_logit": correct_logit,
                "predicted_logit": predicted_logit,
                "max_wrong_choice": LETTER_MAP[int(max_wrong_idx)],
                "max_wrong_logit": max_wrong_logit,
                "correct_vs_predicted_margin": correct_logit - predicted_logit,
                "correct_vs_max_wrong_margin": correct_logit - max_wrong_logit,
            }
        )
    return records


def summarize_answer_trace(layer_answer_trace):
    if not layer_answer_trace:
        return None
    recovery = max(layer_answer_trace, key=lambda item: float(item["correct_vs_max_wrong_margin"]))
    emergence = min(layer_answer_trace, key=lambda item: float(item["correct_vs_max_wrong_margin"]))
    return {
        "logit_recovery_layer": int(recovery["layer"]),
        "logit_emergence_layer": int(emergence["layer"]),
        "best_correct_margin": float(recovery["correct_vs_max_wrong_margin"]),
        "worst_correct_margin": float(emergence["correct_vs_max_wrong_margin"]),
    }


def prepare_prefix_from_image_and_question(model, tokenizer, image_processor, image, question: str, args):
    from Attention.analyze_step_unmask_attention import (
        build_prompt as build_single_prompt,
    )
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    prompt_text = build_single_prompt(question, args.conv_template)
    image = image.convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
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
    prefix_embeds, prefix_input_ids = ret[4], ret[-1][0]
    visual_mask = prefix_input_ids == IMAGE_TOKEN_INDEX
    return prefix_embeds, prefix_input_ids, visual_mask


def prepare_prefix_from_m3cot_doc(model, tokenizer, image_processor, doc, args):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    image = doc["image"].convert("RGB")
    context = build_m3cot_prompt(doc, args.prompt)
    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
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
    prefix_embeds, prefix_input_ids = ret[4], ret[-1][0]
    visual_mask = prefix_input_ids == IMAGE_TOKEN_INDEX
    return context, prompt_text, prefix_embeds, prefix_input_ids, visual_mask


def analyze_prepared_prefix(
    model,
    tokenizer,
    args,
    prefix_embeds,
    prefix_input_ids,
    visual_mask,
    output_dir: Path,
    save_plots: bool,
    answer_trace_config=None,
):
    from Attention.analyze_step_unmask_attention import capture_attention_readonly
    from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch

    output_dir.mkdir(parents=True, exist_ok=True)
    core_model = model.get_model()
    layers_to_capture = list(range(len(core_model.transformer.blocks)))
    _ = prefix_input_ids

    with torch.no_grad():
        past_key_values = core_model(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
        x = torch.full((1, args.max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=args.device)

        if args.max_new_tokens % args.block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length.")

        num_blocks = args.max_new_tokens // args.block_length
        steps = args.max_new_tokens // num_blocks
        if args.step_ratio is not None:
            steps = int(steps * args.step_ratio)
        if steps <= 0:
            raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

        step_records = []
        best_step_idx = None
        best_visual_maps = {}
        best_recovery_score = float("-inf")
        global_step_idx = 0
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

            for local_step_idx in range(num_transfer_tokens.shape[1]):
                mask_index = x == MASK_TOKEN_ID
                block_mask_index = mask_index[:, block_slice]
                if block_mask_index.sum().item() == 0:
                    continue

                current_embeds = core_model.transformer.wte(x)
                if answer_trace_config is None:
                    outputs = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values)
                else:
                    outputs = core_model(
                        None,
                        input_embeddings=current_embeds,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                    )
                logits = outputs.logits
                logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0_p = compute_confidence(logits, x0, args.remasking)

                x0_p[:, (block_idx + 1) * args.block_length :] = -torch.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for batch_idx in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[batch_idx, local_step_idx].item())
                    _, select_index = torch.topk(confidence[batch_idx], k=k)
                    transfer_index[batch_idx, select_index] = True

                selected_queries = torch.where(transfer_index[0])[0]
                layer_answer_trace = None
                answer_trace_summary = None
                if answer_trace_config is not None and selected_queries.numel() > 0:
                    predicted_option_logits = logits[0, selected_queries, :][:, answer_trace_config["option_token_ids"]]
                    predicted_choice_idx = int(predicted_option_logits.mean(dim=0).argmax().item())
                    layer_answer_trace = compute_layer_answer_trace(
                        core_model,
                        outputs.hidden_states,
                        selected_queries=selected_queries,
                        option_token_ids=answer_trace_config["option_token_ids"],
                        correct_choice_idx=int(answer_trace_config["correct_choice_idx"]),
                        predicted_choice_idx=predicted_choice_idx,
                    )
                    answer_trace_summary = summarize_answer_trace(layer_answer_trace)
                    answer_trace_summary["predicted_choice"] = LETTER_MAP[predicted_choice_idx]

                with capture_attention_readonly(model, layers_to_capture, capture_prefill=False, capture_decode=True) as attn_store:
                    _ = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits

                layer_metrics = collect_layer_selection_metrics(
                    attention_store=attn_store,
                    visual_mask=visual_mask,
                    selected_queries=selected_queries,
                    layers=layers_to_capture,
                )
                selection = select_recovery_and_emergence_layers(
                    layer_metrics,
                    min_visual_mass=float(args.min_visual_mass),
                )
                if float(selection["recovery_score"]) > best_recovery_score:
                    best_recovery_score = float(selection["recovery_score"])
                    best_step_idx = global_step_idx
                    best_visual_maps = {}
                    for name, layer_idx in (
                        ("recovery", selection["recovery_layer"]),
                        ("emergence", selection["emergence_layer"]),
                    ):
                        metric = find_metric(layer_metrics, int(layer_idx))
                        best_visual_maps[name] = {
                            "layer": int(layer_idx),
                            "visual_map": metric["visual_map"].detach().cpu().clone(),
                        }

                step_records.append(
                    {
                        "global_step_idx": global_step_idx,
                        "block_idx": block_idx,
                        "local_step_idx": local_step_idx,
                        "selected_queries": [int(idx) for idx in selected_queries.detach().cpu().tolist()],
                        "selection": selection,
                        "layer_metrics": strip_visual_maps(layer_metrics),
                        "answer_trace_summary": answer_trace_summary,
                        "layer_answer_trace": layer_answer_trace,
                    }
                )

                x[transfer_index] = x0[transfer_index]
                global_step_idx += 1
        final_text = tokenizer.batch_decode(x, skip_special_tokens=False)[0].replace("<|endoftext|>", "")

    if not step_records:
        raise RuntimeError("No diffusion steps were recorded.")

    energy_matrix = [
        [item["laplacian_energy"] for item in record["layer_metrics"]]
        for record in step_records
    ]
    mass_matrix = [
        [item["visual_mass"] for item in record["layer_metrics"]]
        for record in step_records
    ]
    score_matrix = [
        [item["selection_score"] for item in record["layer_metrics"]]
        for record in step_records
    ]

    plots = {}
    if save_plots:
        energy_path = output_dir / "layer_step_laplacian_energy_heatmap.png"
        mass_path = output_dir / "layer_step_visual_mass_heatmap.png"
        score_path = output_dir / "layer_step_selection_score_heatmap.png"
        selected_path = output_dir / "selected_layers_over_steps.png"
        selected_score_path = output_dir / "selected_layer_scores_over_steps.png"
        answer_margin_heatmap_path = output_dir / "layer_step_correct_answer_margin_heatmap.png"
        answer_margin_curve_path = output_dir / "correct_answer_margin_by_layer.png"
        save_heatmap(torch.tensor(energy_matrix).transpose(0, 1), energy_path, "Laplacian Energy by Layer and Step", "Laplacian Energy", args.dpi)
        save_heatmap(torch.tensor(mass_matrix).transpose(0, 1), mass_path, "Visual Attention Mass by Layer and Step", "Visual Attention Mass", args.dpi)
        save_heatmap(torch.tensor(score_matrix).transpose(0, 1), score_path, "Layer Selection Score by Layer and Step", "Visual Mass x Laplacian Energy", args.dpi)
        save_selected_layer_curve(step_records, selected_path, args.dpi)
        save_layer_score_curve(step_records, selected_score_path, args.dpi)
        save_answer_margin_heatmap(step_records, answer_margin_heatmap_path, args.dpi)
        save_answer_margin_curve(step_records, answer_margin_curve_path, args.dpi)
        plots.update(
            {
                "laplacian_energy_heatmap": str(energy_path),
                "visual_mass_heatmap": str(mass_path),
                "selection_score_heatmap": str(score_path),
                "selected_layers_over_steps": str(selected_path),
                "selected_layer_scores_over_steps": str(selected_score_path),
            }
        )
        if any(record.get("layer_answer_trace") for record in step_records):
            plots["correct_answer_margin_heatmap"] = str(answer_margin_heatmap_path)
            plots["correct_answer_margin_by_layer"] = str(answer_margin_curve_path)

    visual_map_paths = {}
    best_record = None if best_step_idx is None else step_records[int(best_step_idx)]
    if save_plots:
        for name, payload in best_visual_maps.items():
            layer_idx = int(payload["layer"])
            path = output_dir / f"{name}_layer_{layer_idx}_visual_map_step_{best_step_idx}.png"
            save_visual_map(
                payload["visual_map"],
                path,
                f"{name.title()} Layer {layer_idx} Visual Map at Step {best_step_idx}",
                args.dpi,
            )
            visual_map_paths[name] = str(path)
        plots["representative_visual_maps"] = visual_map_paths
    summary = {
        "final_text": final_text,
        "min_visual_mass": float(args.min_visual_mass),
        "num_steps": len(step_records),
        "plots": plots,
        "representative_step": None if best_record is None else best_record,
        "step_records": step_records,
    }
    return summary


def analyze_layer_selection(model, tokenizer, image_processor, args):
    image = Image.open(args.image).convert("RGB")
    prefix_embeds, prefix_input_ids, visual_mask = prepare_prefix_from_image_and_question(
        model,
        tokenizer,
        image_processor,
        image,
        args.question,
        args,
    )
    output_dir = Path(args.output_dir)
    summary = analyze_prepared_prefix(
        model,
        tokenizer,
        args,
        prefix_embeds=prefix_embeds,
        prefix_input_ids=prefix_input_ids,
        visual_mask=visual_mask,
        output_dir=output_dir,
        save_plots=True,
    )
    summary["image"] = args.image
    summary["question"] = args.question
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(sanitize_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path, summary


def select_dataset_indices(num_items: int, args) -> List[int]:
    start = max(0, int(args.start_index))
    available = list(range(start, num_items))
    limit = min(int(args.limit), len(available))
    if limit <= 0:
        return []
    if args.sample_mode == "sequential":
        return available[:limit]
    rng = random.Random(int(args.sample_seed))
    return sorted(rng.sample(available, limit))


def save_layer_histogram(records, key: str, output_path: Path, title: str, dpi: int):
    layers = [int(record["representative_step"]["selection"][key]) for record in records if record.get("representative_step")]
    if not layers:
        return
    max_layer = max(layers)
    bins = list(range(max_layer + 2))
    plt.figure(figsize=(12, 4.8))
    plt.hist(layers, bins=bins, align="left", rwidth=0.8)
    plt.xlabel("Layer Index")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_average_heatmap(records, metric_name: str, output_path: Path, title: str, colorbar_label: str, dpi: int):
    matrices = []
    for record in records:
        step_records = record.get("step_records", [])
        if not step_records:
            continue
        matrix = [
            [float(item[metric_name]) for item in step["layer_metrics"]]
            for step in step_records
        ]
        matrices.append(torch.tensor(matrix, dtype=torch.float32))
    if not matrices:
        return

    min_steps = min(matrix.shape[0] for matrix in matrices)
    min_layers = min(matrix.shape[1] for matrix in matrices)
    stacked = torch.stack([matrix[:min_steps, :min_layers] for matrix in matrices], dim=0)
    average = stacked.mean(dim=0).transpose(0, 1)
    save_heatmap(average, output_path, title, colorbar_label, dpi)


def save_average_answer_margin_heatmap(records, output_path: Path, dpi: int):
    matrices = []
    for record in records:
        step_records = record.get("step_records", [])
        matrix = []
        for step in step_records:
            trace = step.get("layer_answer_trace")
            if trace:
                matrix.append([float(item["correct_vs_max_wrong_margin"]) for item in trace])
        if matrix:
            matrices.append(torch.tensor(matrix, dtype=torch.float32))
    if not matrices:
        return False
    min_steps = min(matrix.shape[0] for matrix in matrices)
    min_layers = min(matrix.shape[1] for matrix in matrices)
    stacked = torch.stack([matrix[:min_steps, :min_layers] for matrix in matrices], dim=0)
    average = stacked.mean(dim=0).transpose(0, 1)
    save_heatmap(
        average,
        output_path,
        "Average Correct Answer Margin by Layer and Step",
        "Correct Logit - Max Wrong Logit",
        dpi,
        cmap="coolwarm",
    )
    return True


def save_average_answer_margin_curve(records, output_path: Path, dpi: int):
    layer_values = {}
    for record in records:
        for step in record.get("step_records", []):
            trace = step.get("layer_answer_trace")
            if not trace:
                continue
            for item in trace:
                layer_values.setdefault(int(item["layer"]), []).append(float(item["correct_vs_max_wrong_margin"]))
    if not layer_values:
        return False
    layers = sorted(layer_values)
    means = [sum(layer_values[layer]) / len(layer_values[layer]) for layer in layers]
    plt.figure(figsize=(12, 4.8))
    plt.plot(layers, means, linewidth=1.8, marker="o", markersize=3)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Layer Index")
    plt.ylabel("Correct Logit - Max Wrong Logit")
    plt.title("Average Correct Answer Margin Across M3CoT Samples")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return True


def save_answer_trace_layer_histogram(records, key: str, output_path: Path, title: str, dpi: int):
    layers = []
    for record in records:
        for step in record.get("step_records", []):
            summary = step.get("answer_trace_summary")
            if summary is not None:
                layers.append(int(summary[key]))
    if not layers:
        return False
    max_layer = max(layers)
    bins = list(range(max_layer + 2))
    plt.figure(figsize=(12, 4.8))
    plt.hist(layers, bins=bins, align="left", rwidth=0.8)
    plt.xlabel("Layer Index")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return True


def summarize_m3cot_records(sample_summaries, output_dir: Path, args):
    aggregate_dir = output_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    recovery_hist_path = aggregate_dir / "recovery_layer_histogram.png"
    emergence_hist_path = aggregate_dir / "emergence_layer_histogram.png"
    average_energy_path = aggregate_dir / "average_laplacian_energy_heatmap.png"
    average_mass_path = aggregate_dir / "average_visual_mass_heatmap.png"
    average_score_path = aggregate_dir / "average_selection_score_heatmap.png"
    average_answer_margin_path = aggregate_dir / "average_correct_answer_margin_heatmap.png"
    answer_margin_curve_path = aggregate_dir / "average_correct_answer_margin_by_layer.png"
    logit_recovery_hist_path = aggregate_dir / "logit_recovery_layer_histogram.png"
    logit_emergence_hist_path = aggregate_dir / "logit_emergence_layer_histogram.png"

    save_layer_histogram(sample_summaries, "recovery_layer", recovery_hist_path, "Representative Recovery Layers", args.dpi)
    save_layer_histogram(sample_summaries, "emergence_layer", emergence_hist_path, "Representative Emergence Layers", args.dpi)
    save_average_heatmap(sample_summaries, "laplacian_energy", average_energy_path, "Average Laplacian Energy by Layer and Step", "Laplacian Energy", args.dpi)
    save_average_heatmap(sample_summaries, "visual_mass", average_mass_path, "Average Visual Attention Mass by Layer and Step", "Visual Attention Mass", args.dpi)
    save_average_heatmap(sample_summaries, "selection_score", average_score_path, "Average Layer Selection Score by Layer and Step", "Visual Mass x Laplacian Energy", args.dpi)
    has_answer_margin_heatmap = save_average_answer_margin_heatmap(sample_summaries, average_answer_margin_path, args.dpi)
    has_answer_margin_curve = save_average_answer_margin_curve(sample_summaries, answer_margin_curve_path, args.dpi)
    has_logit_recovery_hist = save_answer_trace_layer_histogram(sample_summaries, "logit_recovery_layer", logit_recovery_hist_path, "Logit Recovery Layers Across Steps", args.dpi)
    has_logit_emergence_hist = save_answer_trace_layer_histogram(sample_summaries, "logit_emergence_layer", logit_emergence_hist_path, "Logit Emergence Layers Across Steps", args.dpi)

    representative_layers = []
    for record in sample_summaries:
        representative = record.get("representative_step")
        if representative is None:
            continue
        selection = representative["selection"]
        representative_layers.append(
            {
                "sample_index": int(record["sample_index"]),
                "id": record.get("id"),
                "answer": record.get("answer"),
                "recovery_layer": int(selection["recovery_layer"]),
                "emergence_layer": int(selection["emergence_layer"]),
                "recovery_score": float(selection["recovery_score"]),
                "emergence_score": float(selection["emergence_score"]),
            }
        )

    plots = {
        "recovery_layer_histogram": str(recovery_hist_path),
        "emergence_layer_histogram": str(emergence_hist_path),
        "average_laplacian_energy_heatmap": str(average_energy_path),
        "average_visual_mass_heatmap": str(average_mass_path),
        "average_selection_score_heatmap": str(average_score_path),
    }
    if has_answer_margin_heatmap:
        plots["average_correct_answer_margin_heatmap"] = str(average_answer_margin_path)
    if has_answer_margin_curve:
        plots["average_correct_answer_margin_by_layer"] = str(answer_margin_curve_path)
    if has_logit_recovery_hist:
        plots["logit_recovery_layer_histogram"] = str(logit_recovery_hist_path)
    if has_logit_emergence_hist:
        plots["logit_emergence_layer_histogram"] = str(logit_emergence_hist_path)

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "prompt": args.prompt,
        "trace_answer_logits": bool(args.trace_answer_logits),
        "sample_mode": args.sample_mode,
        "sample_seed": int(args.sample_seed),
        "num_samples": len(sample_summaries),
        "representative_layers": representative_layers,
        "plots": plots,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(sanitize_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path, summary


def analyze_m3cot_layer_selection(model, tokenizer, image_processor, args):
    import datasets
    from tqdm import tqdm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    indices = select_dataset_indices(len(dataset), args)
    if not indices:
        raise ValueError("No M3CoT samples selected. Check --start-index and --limit.")

    sample_summaries = []
    for order_idx, dataset_idx in enumerate(tqdm(indices, desc="M3CoT layer selection")):
        doc = dataset[dataset_idx]
        sample_id = sanitize_filename(doc.get("id", dataset_idx))
        sample_dir = output_dir / "samples" / f"{order_idx:03d}_{dataset_idx}_{sample_id}"
        context, prompt_text, prefix_embeds, prefix_input_ids, visual_mask = prepare_prefix_from_m3cot_doc(
            model,
            tokenizer,
            image_processor,
            doc,
            args,
        )
        answer_trace_config = None
        correct_choice_idx = None
        option_token_ids = None
        if args.trace_answer_logits:
            option_token_ids = get_option_token_ids(tokenizer, len(doc["choices"]))
            correct_choice_idx = answer_to_index(doc["answer"], doc["choices"])
            answer_trace_config = {
                "option_token_ids": option_token_ids,
                "correct_choice_idx": int(correct_choice_idx),
            }
        sample_summary = analyze_prepared_prefix(
            model,
            tokenizer,
            args,
            prefix_embeds=prefix_embeds,
            prefix_input_ids=prefix_input_ids,
            visual_mask=visual_mask,
            output_dir=sample_dir,
            save_plots=bool(args.save_sample_plots),
            answer_trace_config=answer_trace_config,
        )
        sample_summary.update(
            {
                "sample_index": int(dataset_idx),
                "id": doc.get("id"),
                "domain": doc.get("domain"),
                "topic": doc.get("topic"),
                "category": doc.get("category"),
                "question": doc.get("question"),
                "choices": list(doc.get("choices", [])),
                "answer": doc.get("answer"),
                "correct_choice_idx": None if correct_choice_idx is None else int(correct_choice_idx),
                "option_token_ids": option_token_ids,
                "m3cot_prompt": context,
                "full_prompt": prompt_text,
            }
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "summary.json").write_text(json.dumps(sanitize_for_json(sample_summary), ensure_ascii=False, indent=2), encoding="utf-8")
        sample_summaries.append(sample_summary)

    return summarize_m3cot_records(sample_summaries, output_dir, args)


def main():
    args = parse_args()
    from llava.model.builder import load_pretrained_model

    dtype = get_torch_dtype(args.torch_dtype)
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
    model.to(dtype)

    if args.m3cot:
        summary_path, summary = analyze_m3cot_layer_selection(model, tokenizer, image_processor, args)
    else:
        summary_path, summary = analyze_layer_selection(model, tokenizer, image_processor, args)
    print(f"Saved summary to: {summary_path}")
    for name, path in summary["plots"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
