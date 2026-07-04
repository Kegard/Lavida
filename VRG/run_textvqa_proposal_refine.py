"""
这个脚本用于验证并分析下面这类两阶段生成实验是否可行：

1. 先按照原生 LLaDA / mask-predict 生成路径运行到第 k 个 denoising step。
2. 在第 k 步读取“完整答案位置上的 x0 提案”（proposal）。
   这里的“完整 x0”指：虽然当前真实状态 x 里仍有很多 mask，但模型在这一步
   对所有答案位置都给出了一个 top-1 token 预测；我们把这组 top-1 预测拼成
   一整句 proposal。
3. 对 proposal 中每个答案位置计算“是否值得拿去后期 refinement”的优先级。
   当前只保留两种最干净的策略：
   - `confidence`：confidence 越低，越应该 remask；
   - `vis_priority`：cond-uncond 越大，说明该 token 越依赖视觉证据，
     因此越应该 remask。
   - `vis_confidence_priority`：只有当 token 既依赖视觉、又不够确定时，
     才应该优先 remask。其优先级定义为
     `visual_delta * (1 - confidence)`。
4. 按照选定策略，将一部分位置重新置回 mask。
5. 基于这个“部分确定、部分待修正”的 proposal，再执行 r 步短 refinement。

设计动机：

- 你当前的 stepwise x0 实验表明，TextVQA 上很多样本在前中期就已经出现了
  接近正确答案的“词形骨架”或语义线索；
- 但最终 exact match 的明显提升通常集中在最后几步，说明后期 refinement
  对 OCR、拼写、数字补全等表面形式修正仍然重要；
- 更关键的是，如果只看普通 confidence，这件事和单模态工作区分度不高；
- 因此，这个脚本额外提供一组“视觉感知”的 proposal 分析信号，用来回答：
  当前 token 究竟只是语言模型自己很自信，还是它真的得到了视觉证据支持。

和原始 `VRG/run_textvqa_stepwise_x0.py` 的区别：

- 原脚本会把每一个原生 denoising step 的 x0 都记录下来，并逐步评估；
- 本脚本只关心一个指定的 proposal step（`--proposal-step`）；
- 在这个 step 上抽取完整 x0 proposal；
- 然后按照 remask policy 选择一部分位置重新 mask；
- 最后再跑固定的 `--late-refine-steps` 步 refinement。

本脚本当前只支持两类 remask policy：

- `confidence`：只看条件分支自身的 token confidence；
- `vis_priority`：比较有图条件和去视觉条件后的无图分支，用
  `visual_delta = p_cond(x0) - p_uncond(x0)` 度量 token 对视觉条件的依赖性，
  visual_delta 越大，越应该被 remask。
- `vis_confidence_priority`：将视觉相关性和不确定性做乘法结合。
  视觉相关但已经很确定的 token 不再优先 remask；只有“视觉相关且不够确定”的
  token 才会获得更高的 remask 优先级。

注意事项：

- 这里的 refinement 仍然是原生 mask-predict 路径，不涉及 timestep VRG、
  cond/uncond guidance 或 alpha 调度；
- 如果 `--proposal-remask-ratio=0`，则 proposal 之后不会留下任何 mask，
  后续 refinement 也就不会真正更新任何 token；
- 如果 `--late-refine-steps=0`，则该实验退化为“读取第 k 步完整 x0 proposal，
  直接评估它”的 baseline；
- proposal 后的第二阶段 refinement 不再严格复现原始 block schedule，而是用
  “剩余 masked token 在剩余 refinement steps 中均匀分配”的方式完成短修正；
- 当 `remask-policy` 是 `vis_priority` 时，脚本会额外构造一条“去视觉条件”的
  无图前缀，用来分析视觉证据是否真的在支撑当前 token。
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Scale_Attention.reweight_patch import (
    build_prefix_from_multimodal_inputs,
    get_torch_dtype,
    maybe_disable_torch_compile,
)
from VRG.run_textvqa_visual_warmup import (
    build_prompt,
    clean_generated_text,
    compute_textvqa_score,
    construct_textvqa_prompt,
    load_textvqa_split,
    normalize_answers,
    prepare_prefix,
)
from VRG.timestep_vrg import build_unconditional_prefix_embeds
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


MASK_TOKEN_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-stage TextVQA experiment: read a full x0 proposal at step k, "
            "remask low-confidence positions, then run a short late refinement pass."
        )
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="VRG/outputs/textvqa_proposal_refine")
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=None)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random", "entrophy", "margin"],
        help="Native denoising path uses this confidence definition for selecting tokens to unmask.",
    )
    parser.add_argument(
        "--proposal-step",
        type=int,
        required=True,
        help="Read the full x0 proposal at this 1-based native denoising step.",
    )
    parser.add_argument(
        "--proposal-remask-ratio",
        type=float,
        default=0.25,
        help="After reading the full proposal, remask this ratio of lowest-confidence answer positions.",
    )
    parser.add_argument(
        "--late-refine-steps",
        type=int,
        default=4,
        help="How many short refinement steps to run after remasking low-confidence positions.",
    )
    parser.add_argument(
        "--remask-policy",
        default="confidence",
        choices=["confidence", "vis_priority", "vis_confidence_priority"],
        help=(
            "How to choose which proposal tokens should be remasked before refinement. "
            "`confidence`: lower confidence -> remask. "
            "`vis_priority`: larger cond-uncond visual delta -> remask. "
            "`vis_confidence_priority`: larger visual_delta * (1 - confidence) -> remask."
        ),
    )
    parser.add_argument(
        "--null-visual-mode",
        default="zeros",
        choices=["zeros", "mask_token"],
        help="How to remove visual evidence when computing visual-aware proposal scores.",
    )
    parser.add_argument(
        "--refine-guidance",
        default="none",
        choices=["none", "vcd"],
        help="Optional logits guidance used only during the late refinement stage.",
    )
    parser.add_argument(
        "--refine-weak-visual-mode",
        default="diffusion_noise",
        choices=["null_visual", "diffusion_noise"],
        help="Weak visual condition for VCD-guided refinement logits.",
    )
    parser.add_argument(
        "--vcd-refine-alpha",
        type=float,
        default=0.5,
        help="Alpha in guided refinement logits: (1 + alpha) * logits(image) - alpha * logits(weak_visual).",
    )
    parser.add_argument(
        "--refine-guidance-steps",
        type=int,
        default=None,
        help="Apply refinement guidance only on the first k late-refine steps. If unset, use all refine steps.",
    )
    parser.add_argument(
        "--vcd-noise-step",
        type=int,
        default=500,
        help="Forward-diffusion timestep for --refine-weak-visual-mode diffusion_noise.",
    )
    parser.add_argument(
        "--vcd-noise-seed",
        type=int,
        default=42,
        help="Optional noise seed for --refine-weak-visual-mode diffusion_noise.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def compute_remasking_confidence(logits, x0, remasking):
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        log_probs = torch.log(probs + epsilon)
        return torch.sum(probs * log_probs, dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise NotImplementedError(remasking)


def decode_answer_tokens(tokenizer, token_ids):
    return clean_generated_text(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def resolve_steps_per_block(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens
    if steps % num_blocks != 0 and step_per_block is None:
        raise ValueError("Native generation requires steps % num_blocks == 0 unless step_per_block is set.")
    steps = steps // num_blocks

    if step_per_block is None and step_ratio is None:
        step_per_block = block_length
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both --step-per-block and --step-ratio.")
        steps = min(int(step_per_block), block_length)
    if step_ratio:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")
    return num_blocks, steps


def build_proposal_state(proposal_answer, remasked_answer_positions, prefix_length, device):
    answer_length = proposal_answer.shape[0]
    proposal_state = proposal_answer.clone()
    if remasked_answer_positions:
        proposal_state[torch.tensor(remasked_answer_positions, device=device, dtype=torch.long)] = MASK_TOKEN_ID

    x_refine = torch.full((1, prefix_length + answer_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x_refine[:, :prefix_length] = 0
    x_refine[0, prefix_length:] = proposal_state
    return x_refine


def add_diffusion_noise_tensor(image_tensor, noise_step, seed=None):
    if not 0 <= int(noise_step) < 1000:
        raise ValueError("--vcd-noise-step must be in [0, 999].")

    device = image_tensor.device
    dtype = image_tensor.dtype
    betas = torch.linspace(-6, 6, 1000, device=device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)[int(noise_step)]

    if seed is None:
        noise = torch.randn_like(image_tensor)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noise = torch.randn(image_tensor.shape, generator=generator, device=device, dtype=dtype)
    return alpha_bar.sqrt().to(dtype) * image_tensor + (1.0 - alpha_bar).sqrt().to(dtype) * noise


def add_diffusion_noise(images, noise_step, seed=None):
    if isinstance(images, list):
        return [
            add_diffusion_noise_tensor(
                image,
                noise_step=noise_step,
                seed=None if seed is None else int(seed) + idx,
            )
            for idx, image in enumerate(images)
        ]
    return add_diffusion_noise_tensor(images, noise_step=noise_step, seed=seed)


def build_diffusion_noise_prefix(args, model, tokenizer, image_processor, doc):
    image = doc["image"].convert("RGB")
    context = construct_textvqa_prompt(
        doc,
        prompt_mode=getattr(args, "prompt_mode", "auto"),
        pretrained_path=getattr(args, "pretrained", None),
    )
    prompt = build_prompt(context, args.conv_template)

    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)
    noisy_image_tensor = add_diffusion_noise(
        image_tensor,
        noise_step=args.vcd_noise_step,
        seed=args.vcd_noise_seed,
    )

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    weak_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=noisy_image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return weak_prefix_embeds


def compute_visual_token_metrics(logits_cond, logits_uncond, x0):
    """计算 proposal token 的条件/无图分支差异，用于分析视觉证据是否真正起作用。"""
    cond_probs = F.softmax(logits_cond.to(torch.float64), dim=-1)
    uncond_probs = F.softmax(logits_uncond.to(torch.float64), dim=-1)

    cond_token_prob = torch.gather(cond_probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    uncond_token_prob = torch.gather(uncond_probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

    cond_entropy = -(cond_probs * torch.log(cond_probs + 1e-10)).sum(dim=-1)
    uncond_entropy = -(uncond_probs * torch.log(uncond_probs + 1e-10)).sum(dim=-1)

    logit_delta = torch.gather((logits_cond - logits_uncond).to(torch.float64), dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    visual_delta = cond_token_prob - uncond_token_prob
    visual_entropy_gap = uncond_entropy - cond_entropy
    return {
        "cond_token_prob": cond_token_prob,
        "uncond_token_prob": uncond_token_prob,
        "token_logit_delta": logit_delta,
        "visual_delta": visual_delta,
        "cond_entropy": cond_entropy,
        "uncond_entropy": uncond_entropy,
        "visual_entropy_gap": visual_entropy_gap,
    }


def build_remask_priority(base_confidence, visual_metrics, remask_policy):
    """构造 remask 优先级。优先级越大，越应该被重新 mask 并进入 refinement。"""
    if remask_policy == "confidence":
        return 1.0 - base_confidence
    if remask_policy == "vis_priority":
        if visual_metrics is None:
            raise ValueError("visual_metrics is required for vis_priority remasking.")
        return visual_metrics["visual_delta"]
    if remask_policy == "vis_confidence_priority":
        if visual_metrics is None:
            raise ValueError("visual_metrics is required for vis_confidence_priority remasking.")
        return visual_metrics["visual_delta"] * (1.0 - base_confidence)
    raise ValueError(f"Unsupported remask policy: {remask_policy}")


def _max_softmax_probability(logits):
    logits = logits.to(torch.float64)
    return torch.exp(logits.max(dim=-1).values - torch.logsumexp(logits, dim=-1))


def _apply_confidence_gate(cond_logits, guided_logits, answer_slice, tau):
    cond_ans = cond_logits[:, answer_slice]
    guided_ans = guided_logits[:, answer_slice]
    use_guided = (
        _max_softmax_probability(guided_ans)
        - _max_softmax_probability(cond_ans)
    ) > float(tau)
    merged = cond_logits.clone()
    merged[:, answer_slice] = torch.where(
        use_guided.unsqueeze(-1), guided_ans, cond_ans,
    )
    return merged


def _forward_with_prefix_cache(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=None):
    answer_embeds = core_model.transformer.wte(x[:, prefix_length:])
    if prefix_kv_cache is None:
        output = core_model(None, input_embeddings=prefix_embeds, use_cache=True)
        new_cache = output.attn_key_values
        answer_output = core_model(None, input_embeddings=answer_embeds, past_key_values=new_cache)
        full_logits = torch.zeros(x.shape[0], prefix_length + answer_embeds.shape[1], answer_output.logits.shape[-1], dtype=answer_output.logits.dtype, device=answer_output.logits.device)
        full_logits[:, prefix_length:] = answer_output.logits
        return full_logits, new_cache
    output = core_model(None, input_embeddings=answer_embeds, past_key_values=prefix_kv_cache)
    full_logits = torch.zeros(x.shape[0], prefix_length + answer_embeds.shape[1], output.logits.shape[-1], dtype=output.logits.dtype, device=output.logits.device)
    full_logits[:, prefix_length:] = output.logits
    return full_logits, prefix_kv_cache


@torch.no_grad()
def run_proposal_then_refine(
    core_model,
    tokenizer,
    prefix_embeds,
    prefix_input_ids_full,
    max_new_tokens,
    block_length,
    step_per_block,
    cfg_scale,
    temperature,
    remasking,
    schedule,
    schedule_shift,
    step_ratio,
    proposal_step,
    proposal_remask_ratio,
    late_refine_steps,
    remask_policy="confidence",
    null_visual_mode="zeros",
    refine_guidance="none",
    refine_weak_visual_mode="diffusion_noise",
    refine_weak_prefix_embeds=None,
    vcd_refine_alpha=0.5,
    refine_guidance_steps=None,
    vcd_noise_step=500,
    vcd_noise_seed=42,
    refine_confidence_gate_tau=None,
    use_prefix_cache=True,
):
    if cfg_scale > 0.0:
        raise NotImplementedError("cfg_scale > 0.0 is not supported in the native path.")
    if proposal_step <= 0:
        raise ValueError("--proposal-step must be >= 1.")
    if not 0.0 <= proposal_remask_ratio <= 1.0:
        raise ValueError("--proposal-remask-ratio must be within [0, 1].")
    if late_refine_steps < 0:
        raise ValueError("--late-refine-steps must be >= 0.")
    if refine_guidance_steps is not None and refine_guidance_steps <= 0:
        raise ValueError("--refine-guidance-steps must be > 0 when set.")
    need_visual_analysis = remask_policy in {"vis_priority", "vis_confidence_priority"}
    if need_visual_analysis and prefix_input_ids_full is None:
        raise ValueError("prefix_input_ids_full is required for visual-aware remasking.")
    if refine_guidance == "vcd":
        if refine_weak_visual_mode == "null_visual" and prefix_input_ids_full is None:
            raise ValueError("prefix_input_ids_full is required for null-visual VCD refinement.")
        if refine_weak_visual_mode == "diffusion_noise" and refine_weak_prefix_embeds is None:
            raise ValueError("refine_weak_prefix_embeds is required for diffusion-noise VCD refinement.")
    elif refine_guidance != "none":
        raise ValueError(f"Unsupported refine guidance: {refine_guidance}")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    num_blocks, steps_per_block = resolve_steps_per_block(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )

    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0

    schedule_value = None if schedule == "none" else schedule
    schedule_kwargs = {"shift": schedule_shift} if schedule_value == "shift" else None

    proposal_trace = []
    global_step = 0
    proposal_payload = None
    uncond_prefix_embeds = None
    if need_visual_analysis:
        uncond_prefix_embeds, _ = build_unconditional_prefix_embeds(
            core_model=core_model,
            prefix_embeds=prefix_embeds,
            prefix_input_ids_full=prefix_input_ids_full,
            null_visual_mode=null_visual_mode,
        )

    refine_uncond_prefix_embeds = None
    if refine_guidance == "vcd":
        if refine_weak_visual_mode == "null_visual":
            refine_uncond_prefix_embeds, _ = build_unconditional_prefix_embeds(
                core_model=core_model,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                null_visual_mode=null_visual_mode,
            )
        elif refine_weak_visual_mode == "diffusion_noise":
            refine_uncond_prefix_embeds = refine_weak_prefix_embeds
        else:
            raise ValueError(f"Unsupported refine weak visual mode: {refine_weak_visual_mode}")

    cond_prefix_kv = None
    uncond_prefix_kv = None
    refine_weak_prefix_kv = None

    for block_idx in range(num_blocks):
        block_start = prefix_length + block_idx * block_length
        block_end = prefix_length + (block_idx + 1) * block_length
        block_slice = slice(block_start, block_end)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps_per_block,
            schedule=schedule_value,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            if use_prefix_cache:
                logits, cond_prefix_kv = _forward_with_prefix_cache(
                    core_model, x, prefix_embeds, prefix_length, cond_prefix_kv,
                )
            else:
                current_embeds = core_model.transformer.wte(x)
                current_embeds[:, :prefix_length] = prefix_embeds
                logits = core_model(None, input_embeddings=current_embeds).logits
            logits_uncond = None
            visual_metrics = None
            if need_visual_analysis:
                if use_prefix_cache:
                    logits_uncond, uncond_prefix_kv = _forward_with_prefix_cache(
                        core_model, x, uncond_prefix_embeds, prefix_length, uncond_prefix_kv,
                    )
                else:
                    current_embeds_uncond = core_model.transformer.wte(x)
                    current_embeds_uncond[:, :prefix_length] = uncond_prefix_embeds
                    logits_uncond = core_model(None, input_embeddings=current_embeds_uncond).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            raw_confidence = compute_remasking_confidence(logits, x0, remasking)
            if need_visual_analysis:
                visual_metrics = compute_visual_token_metrics(
                    logits_cond=logits,
                    logits_uncond=logits_uncond,
                    x0=x0,
                )
            x0_p = raw_confidence.clone()
            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)

            proposal_answer = x0[0, prefix_length:].detach().clone()
            proposal_confidence = raw_confidence[0, prefix_length:].detach().clone()
            remask_priority = build_remask_priority(
                base_confidence=raw_confidence,
                visual_metrics=visual_metrics,
                remask_policy=remask_policy,
            )[0, prefix_length:].detach().clone()
            proposal_text = decode_answer_tokens(tokenizer, proposal_answer.detach().cpu().tolist())

            confidence = torch.where(mask_index, x0_p, -torch.inf)
            k = int(num_transfer_tokens[0, step_idx].item())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

            global_step += 1
            proposal_trace.append(
                {
                    "step": int(global_step),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "num_transferred": int(k),
                    "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                    "num_masked_after_step": int((x[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
                    "candidate_text": proposal_text,
                }
            )

            if global_step == proposal_step:
                proposal_payload = {
                    "proposal_answer": proposal_answer,
                    "proposal_confidence": proposal_confidence,
                    "remask_priority": remask_priority,
                    "proposal_text": proposal_text,
                    "proposal_trace": list(proposal_trace),
                    "steps_per_block": int(steps_per_block),
                    "num_blocks": int(num_blocks),
                    "total_native_steps": int(num_blocks * steps_per_block),
                }
                if visual_metrics is not None:
                    proposal_payload["visual_metrics"] = {
                        "cond_token_prob": visual_metrics["cond_token_prob"][0, prefix_length:].detach().clone(),
                        "uncond_token_prob": visual_metrics["uncond_token_prob"][0, prefix_length:].detach().clone(),
                        "token_logit_delta": visual_metrics["token_logit_delta"][0, prefix_length:].detach().clone(),
                        "visual_delta": visual_metrics["visual_delta"][0, prefix_length:].detach().clone(),
                        "cond_entropy": visual_metrics["cond_entropy"][0, prefix_length:].detach().clone(),
                        "uncond_entropy": visual_metrics["uncond_entropy"][0, prefix_length:].detach().clone(),
                        "visual_entropy_gap": visual_metrics["visual_entropy_gap"][0, prefix_length:].detach().clone(),
                    }
                break

        if proposal_payload is not None:
            break

    if proposal_payload is None:
        raise ValueError(
            f"--proposal-step={proposal_step} exceeds available native denoising steps "
            f"({num_blocks * steps_per_block})."
        )

    proposal_answer = proposal_payload["proposal_answer"]
    proposal_confidence = proposal_payload["proposal_confidence"]
    remask_priority = proposal_payload["remask_priority"]
    proposal_text = proposal_payload["proposal_text"]
    visual_metrics_payload = proposal_payload.get("visual_metrics")

    num_answer_positions = int(proposal_answer.shape[0])
    num_to_remask = int(math.floor(num_answer_positions * proposal_remask_ratio))
    if proposal_remask_ratio > 0.0 and num_to_remask == 0:
        num_to_remask = 1

    remasked_answer_positions = []
    if num_to_remask > 0:
        remask_indices = torch.topk(remask_priority, k=num_to_remask, largest=True).indices
        remasked_answer_positions = sorted(int(idx) for idx in remask_indices.detach().cpu().tolist())

    x_refine = build_proposal_state(
        proposal_answer=proposal_answer,
        remasked_answer_positions=remasked_answer_positions,
        prefix_length=prefix_length,
        device=device,
    )

    remasked_position_details = []
    for answer_pos in remasked_answer_positions:
        token_id = int(proposal_answer[answer_pos].item())
        remasked_position_details.append(
            {
                "answer_position": answer_pos,
                "sequence_position": int(prefix_length + answer_pos),
                "proposal_token_id": token_id,
                "proposal_token_text": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "proposal_confidence": float(proposal_confidence[answer_pos].item()),
                "remask_priority": float(remask_priority[answer_pos].item()),
            }
        )
        if visual_metrics_payload is not None:
            remasked_position_details[-1].update(
                {
                    "cond_token_prob": float(visual_metrics_payload["cond_token_prob"][answer_pos].item()),
                    "uncond_token_prob": float(visual_metrics_payload["uncond_token_prob"][answer_pos].item()),
                    "token_logit_delta": float(visual_metrics_payload["token_logit_delta"][answer_pos].item()),
                    "visual_delta": float(visual_metrics_payload["visual_delta"][answer_pos].item()),
                    "cond_entropy": float(visual_metrics_payload["cond_entropy"][answer_pos].item()),
                    "uncond_entropy": float(visual_metrics_payload["uncond_entropy"][answer_pos].item()),
                    "visual_entropy_gap": float(visual_metrics_payload["visual_entropy_gap"][answer_pos].item()),
                }
            )

    refine_records = []
    for refine_step in range(1, late_refine_steps + 1):
        answer_mask = x_refine[:, prefix_length:] == MASK_TOKEN_ID
        masked_remaining = int(answer_mask.sum().item())
        if masked_remaining == 0:
            break

        if use_prefix_cache:
            logits, cond_prefix_kv = _forward_with_prefix_cache(
                core_model, x_refine, prefix_embeds, prefix_length, cond_prefix_kv,
            )
        else:
            current_embeds = core_model.transformer.wte(x_refine)
            current_embeds[:, :prefix_length] = prefix_embeds
            logits = core_model(None, input_embeddings=current_embeds).logits
        guidance_used = (
            refine_guidance == "vcd"
            and (
                refine_guidance_steps is None
                or refine_step <= int(refine_guidance_steps)
            )
        )
        if guidance_used:
            cond_logits = logits
            if use_prefix_cache:
                weak_logits, refine_weak_prefix_kv = _forward_with_prefix_cache(
                    core_model, x_refine, refine_uncond_prefix_embeds, prefix_length, refine_weak_prefix_kv,
                )
            else:
                weak_embeds = core_model.transformer.wte(x_refine)
                weak_embeds[:, :prefix_length] = refine_uncond_prefix_embeds
                weak_logits = core_model(None, input_embeddings=weak_embeds).logits
            guided_logits = (1.0 + float(vcd_refine_alpha)) * cond_logits - float(vcd_refine_alpha) * weak_logits
            if refine_confidence_gate_tau is not None:
                answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
                logits = _apply_confidence_gate(
                    cond_logits, guided_logits, answer_slice, refine_confidence_gate_tau,
                )
            else:
                logits = guided_logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        x0 = torch.where(x_refine == MASK_TOKEN_ID, x0, x_refine)
        confidence = compute_remasking_confidence(logits, x0, remasking)
        confidence = torch.where(x_refine == MASK_TOKEN_ID, confidence, -torch.inf)

        remaining_steps = late_refine_steps - refine_step + 1
        k = int(math.ceil(masked_remaining / remaining_steps))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        _, select_index = torch.topk(confidence[0], k=k)
        transfer_index[0, select_index] = True
        x_refine[transfer_index] = x0[transfer_index]

        candidate_text = decode_answer_tokens(
            tokenizer,
            x0[0, prefix_length:].detach().cpu().tolist(),
        )
        refine_state_text = decode_answer_tokens(
            tokenizer,
            x_refine[0, prefix_length:].detach().cpu().tolist(),
        )
        refine_records.append(
            {
                "refine_step": int(refine_step),
                "num_transferred": int(k),
                "masked_before_step": int(masked_remaining),
                "masked_after_step": int((x_refine[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
                "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                "guidance_used": bool(guidance_used),
                "candidate_text": candidate_text,
                "refine_state_text": refine_state_text,
            }
        )

    final_answer_ids = x_refine[0, prefix_length:].detach().cpu().tolist()
    final_text = decode_answer_tokens(tokenizer, final_answer_ids)
    meta = {
        "prefix_length": int(prefix_length),
        "num_blocks": proposal_payload["num_blocks"],
        "steps_per_block": proposal_payload["steps_per_block"],
        "total_native_steps": proposal_payload["total_native_steps"],
        "proposal_step": int(proposal_step),
        "proposal_remask_ratio": float(proposal_remask_ratio),
        "remask_policy": remask_policy,
        "num_remasked_positions": int(len(remasked_answer_positions)),
        "late_refine_steps_requested": int(late_refine_steps),
        "late_refine_steps_run": int(len(refine_records)),
        "refine_guidance": refine_guidance,
        "refine_weak_visual_mode": refine_weak_visual_mode if refine_guidance == "vcd" else None,
        "vcd_refine_alpha": float(vcd_refine_alpha) if refine_guidance == "vcd" else None,
        "refine_guidance_steps": int(refine_guidance_steps) if refine_guidance_steps is not None else None,
        "refine_confidence_gate_tau": float(refine_confidence_gate_tau) if refine_confidence_gate_tau is not None else None,
        "vcd_noise_step": (
            int(vcd_noise_step)
            if refine_guidance == "vcd" and refine_weak_visual_mode == "diffusion_noise"
            else None
        ),
        "vcd_noise_seed": (
            int(vcd_noise_seed)
            if refine_guidance == "vcd"
            and refine_weak_visual_mode == "diffusion_noise"
            and vcd_noise_seed is not None
            else None
        ),
    }
    output = {
        "proposal_text": proposal_text,
        "proposal_trace": proposal_payload["proposal_trace"],
        "proposal_confidence": proposal_confidence.detach().cpu().tolist(),
        "remask_priority": remask_priority.detach().cpu().tolist(),
        "proposal_answer_ids": proposal_answer.detach().cpu().tolist(),
        "remasked_positions": remasked_position_details,
        "refine_records": refine_records,
        "final_text": final_text,
        "meta": meta,
    }
    if visual_metrics_payload is not None:
        output["proposal_visual_metrics"] = {
            key: value.detach().cpu().tolist()
            for key, value in visual_metrics_payload.items()
        }
    return output


def main():
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    restore_compile = maybe_disable_torch_compile()

    from llava.model.builder import load_pretrained_model

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
    model.to(get_torch_dtype(args.torch_dtype))
    core_model = model.get_model()

    dataset = load_textvqa_split(args.dataset_path, args.dataset_name, args.split)
    answer_processor = EvalAIAnswerProcessor()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    total_elapsed = 0.0
    written = 0
    proposal_score_total = 0.0
    final_score_total = 0.0
    improved_after_refine = 0
    worsened_after_refine = 0
    unchanged_after_refine = 0
    refine_step_totals = defaultdict(float)
    refine_step_counts = defaultdict(int)

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break
            doc = dataset[dataset_index]
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, prefix_input_ids_full = prepare_prefix(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )
            refine_weak_prefix_embeds = None
            if args.refine_guidance == "vcd" and args.refine_weak_visual_mode == "diffusion_noise":
                refine_weak_prefix_embeds = build_diffusion_noise_prefix(
                    args,
                    model,
                    tokenizer,
                    image_processor,
                    doc,
                )
            normalized_answers = normalize_answers(doc, answer_processor)
            run_output = run_proposal_then_refine(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                refine_weak_prefix_embeds=refine_weak_prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=args.schedule_shift,
                step_ratio=args.step_ratio,
                proposal_step=args.proposal_step,
                proposal_remask_ratio=args.proposal_remask_ratio,
                late_refine_steps=args.late_refine_steps,
                remask_policy=args.remask_policy,
                null_visual_mode=args.null_visual_mode,
                refine_guidance=args.refine_guidance,
                refine_weak_visual_mode=args.refine_weak_visual_mode,
                vcd_refine_alpha=args.vcd_refine_alpha,
                refine_guidance_steps=args.refine_guidance_steps,
                vcd_noise_step=args.vcd_noise_step,
                vcd_noise_seed=args.vcd_noise_seed,
            )

            proposal_score, proposal_prediction = compute_textvqa_score(
                normalized_answers,
                run_output["proposal_text"],
                answer_processor,
            )
            final_score, final_prediction = compute_textvqa_score(
                normalized_answers,
                run_output["final_text"],
                answer_processor,
            )
            for refine_record in run_output["refine_records"]:
                step_score, step_prediction = compute_textvqa_score(
                    normalized_answers,
                    refine_record["refine_state_text"],
                    answer_processor,
                )
                refine_record["normalized_prediction"] = step_prediction
                refine_record["exact_match"] = step_score
                step = int(refine_record["refine_step"])
                refine_step_totals[step] += step_score
                refine_step_counts[step] += 1

            elapsed = time.time() - t0
            total_elapsed += elapsed
            proposal_score_total += proposal_score
            final_score_total += final_score

            if final_score > proposal_score:
                improved_after_refine += 1
            elif final_score < proposal_score:
                worsened_after_refine += 1
            else:
                unchanged_after_refine += 1

            record = {
                "dataset_index": int(dataset_index),
                "question_id": doc.get("question_id"),
                "question": context,
                "answers": doc.get("answers"),
                "normalized_answers": normalized_answers,
                "ocr_tokens": doc.get("ocr_tokens"),
                "prompt": prompt,
                "elapsed_sec": elapsed,
                "proposal_text": run_output["proposal_text"],
                "proposal_prediction": proposal_prediction,
                "proposal_exact_match": proposal_score,
                "final_text": run_output["final_text"],
                "final_prediction": final_prediction,
                "final_exact_match": final_score,
                "proposal_answer_ids": run_output["proposal_answer_ids"],
                "proposal_confidence": run_output["proposal_confidence"],
                "remask_priority": run_output["remask_priority"],
                "remasked_positions": run_output["remasked_positions"],
                "proposal_trace": run_output["proposal_trace"],
                "refine_records": run_output["refine_records"],
                "meta": run_output["meta"],
            }
            if "proposal_visual_metrics" in run_output:
                record["proposal_visual_metrics"] = run_output["proposal_visual_metrics"]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} "
                    f"proposal={proposal_score:.3f} final={final_score:.3f} "
                    f"remasked={len(run_output['remasked_positions'])} elapsed={elapsed:.2f}s"
                )

    refine_step_summary = []
    for step in sorted(refine_step_counts):
        count = refine_step_counts[step]
        refine_step_summary.append(
            {
                "refine_step": int(step),
                "mean_exact_match": refine_step_totals[step] / count if count else None,
                "count": int(count),
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "proposal_definition": (
            "Read the full answer-position x0 at proposal_step, then choose remasked positions "
            "either by low confidence or by high visual cond-uncond priority, "
            "then run a short refinement pass."
        ),
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "temperature": args.temperature,
        },
        "proposal_refine": {
            "proposal_step": args.proposal_step,
            "proposal_remask_ratio": args.proposal_remask_ratio,
            "late_refine_steps": args.late_refine_steps,
            "remask_policy": args.remask_policy,
            "null_visual_mode": args.null_visual_mode,
            "refine_guidance": args.refine_guidance,
            "refine_weak_visual_mode": args.refine_weak_visual_mode if args.refine_guidance == "vcd" else None,
            "vcd_refine_alpha": args.vcd_refine_alpha if args.refine_guidance == "vcd" else None,
            "refine_guidance_steps": args.refine_guidance_steps,
            "vcd_noise_step": (
                args.vcd_noise_step
                if args.refine_guidance == "vcd" and args.refine_weak_visual_mode == "diffusion_noise"
                else None
            ),
            "vcd_noise_seed": (
                args.vcd_noise_seed
                if args.refine_guidance == "vcd"
                and args.refine_weak_visual_mode == "diffusion_noise"
                and args.vcd_noise_seed is not None
                else None
            ),
        },
        "proposal_mean_exact_match": proposal_score_total / written if written else None,
        "final_mean_exact_match": final_score_total / written if written else None,
        "mean_gain_from_refine": (final_score_total - proposal_score_total) / written if written else None,
        "num_improved_after_refine": improved_after_refine,
        "num_worsened_after_refine": worsened_after_refine,
        "num_unchanged_after_refine": unchanged_after_refine,
        "refine_step_summary": refine_step_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")
    print(
        "Mean EM: "
        f"proposal={summary['proposal_mean_exact_match']:.4f}, "
        f"final={summary['final_mean_exact_match']:.4f}, "
        f"gain={summary['mean_gain_from_refine']:.4f}"
    )
    restore_compile()


if __name__ == "__main__":
    main()
