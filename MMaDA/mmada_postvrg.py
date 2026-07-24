#!/usr/bin/env python
"""Run MMaDA on PostVRG-style multiple-choice benchmarks.

This runner mirrors the output contract of ``postvrg_final.py`` while using the
official MMaDA multimodal-understanding path:

  image -> MAGVITv2 codes -> <|mmu|><|soi|> image <|eoi|> chat prompt

The default ``--mode baseline`` calls ``MMadaModelLM.mmu_generate`` directly.
``--mode postvrg`` uses the same low-confidence draft/remask/refill idea as
PostVRG, adapted to MMaDA's masked-diffusion sequence interface.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MMADA_ROOT_DEFAULT = Path(__file__).resolve().parent
REPO_ROOT = MMADA_ROOT_DEFAULT.parent
for path in (MMADA_ROOT_DEFAULT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from postvrg_dataset_adapters import add_dataset_adapter_args, load_postvrg_dataset


SPECIAL_TOKENS = (
    "<|soi|>",
    "<|eoi|>",
    "<|sov|>",
    "<|eov|>",
    "<|t2i|>",
    "<|mmu|>",
    "<|t2v|>",
    "<|v2v|>",
    "<|lvg|>",
)
DEFAULT_MASK_TOKEN_ID = 126336
LETTER_MAP = "ABCDEFG"


def extract_answer(text, choices):
    valid_letters = LETTER_MAP[: len(choices)]
    scoped_text = str(text or "")
    if "[Answer]" in scoped_text:
        scoped_text = (
            scoped_text.split("[Answer]")[-1]
            .split("[Rationale]")[0]
            .split("[Context]")[0]
        )

    patterns = [
        rf"Answer\s*:\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b",
        rf"\\boxed\s*\{{?\s*([{valid_letters}{valid_letters.lower()}])\s*\}}?",
        rf"\[Answer\]\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b",
        rf"\(([{valid_letters}{valid_letters.lower()}])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, scoped_text)
        if matches:
            return matches[-1].upper()

    matches = []
    for idx, choice in enumerate(choices):
        if str(choice).lower() in scoped_text.lower():
            matches.append(LETTER_MAP[idx])
    if matches:
        return matches[-1]

    normalized_text = re.sub(r"[\n.,!?]", " ", scoped_text)
    tokens = normalized_text.split()
    matches = []
    for idx, _ in enumerate(choices):
        if LETTER_MAP[idx] in tokens or LETTER_MAP[idx].lower() in tokens:
            matches.append(LETTER_MAP[idx])
    if matches:
        return matches[-1]
    return "FAILED"


def judge_answer(text, choices, answer):
    if isinstance(answer, int):
        answer = LETTER_MAP[answer]
    return extract_answer(text, choices) == answer


def build_choice_block(choices):
    return "\n".join(f"{LETTER_MAP[i]}. {choice}" for i, choice in enumerate(choices))


def build_base_prompt(doc):
    parts = []
    context = (doc.get("context") or "").strip()
    if context:
        parts.append(context)
    parts.append(doc["question"])
    parts.append(build_choice_block(doc["choices"]))
    return "\n".join(parts)


def build_prompt(doc, prompt_style):
    base = build_base_prompt(doc)
    if prompt_style == "direct":
        return base + "\n\nAnswer with the option's letter from the given choices directly."
    if prompt_style == "cot":
        return (
            base
            + "\n\nYou should first think about the reasoning process in the mind and then "
            + "provide the user with the answer. The reasoning process is enclosed within "
            + "<think> </think> tags, i.e. <think> reasoning process here </think> answer here. "
            + "The final answer should use the format Answer: <option letter>."
        )
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
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def clean_generated_text(text):
    text = text.lstrip("!")
    for token in ("<|endoftext|>", "<|eot_id|>", "<|im_end|>", "<|eot|>"):
        text = text.replace(token + "\n", "")
        text = text.replace(token, "")
    return text.strip()


def cli_value(flag, default):
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def add_default(flag, value):
    if value is None:
        return
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def apply_mmada_defaults():
    sample_seed = cli_value("--sample-seed", "42")
    limit = cli_value("--limit", "400")
    mode = cli_value("--mode", "baseline")
    model_tag = Path(cli_value("--model-path", "weight/mmada/MMaDA-8B-MixCoT")).name
    default_output = (
        "M3CoT/PostVRG/outputs/"
        f"mmada_{mode}_{model_tag}_seed{sample_seed}_n{limit}"
    )
    defaults = {
        "--prompt": "cot",
        "--limit": limit,
        "--sample-mode": "random",
        "--sample-seed": sample_seed,
        "--max-new-tokens": 64,
        "--steps": 32,
        "--block-length": 64,
        "--draft-steps": 16,
        "--postmask-steps": 16,
        "--fixed-set-size": 32,
        "--fixed-refill-per-step": 2,
        "--output-dir": default_output,
    }
    for flag, value in defaults.items():
        add_default(flag, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MMaDA with PostVRG-compatible datasets and outputs."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test")
    add_dataset_adapter_args(parser)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--domain-filter", default=None)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/PostVRG/outputs/mmada")

    parser.add_argument("--mmada-root", default=str(MMADA_ROOT_DEFAULT))
    parser.add_argument("--model-path", default="weight/mmada/MMaDA-8B-MixCoT")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--vq-model-path", default="weight/mmada/magvitv2")
    parser.add_argument("--vq-model-type", default="magvitv2", choices=["magvitv2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--image-transform",
        default="center_crop",
        choices=["center_crop", "squash"],
        help="MMaDA image preprocessing. Official examples mostly use center_crop.",
    )

    parser.add_argument("--mode", default="baseline", choices=["baseline", "postvrg"])
    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--max-text-len", type=int, default=2048)
    parser.add_argument(
        "--chat-template",
        default="tokenizer",
        choices=["tokenizer", "manual"],
        help="Use tokenizer.apply_chat_template when available, or a Llama-3-style manual template.",
    )

    parser.add_argument("--draft-steps", type=int, default=None)
    parser.add_argument("--postmask-steps", type=int, default=None)
    parser.add_argument("--fixed-set-size", type=int, default=None)
    parser.add_argument("--fixed-refill-per-step", type=int, default=None)
    parser.add_argument("--refill-confidence-gate", action="store_true")
    parser.add_argument("--draft-visual-mode", default="full", choices=["full", "edge_noise", "random_noise"])
    parser.add_argument("--refine-visual-mode", default="full", choices=["full", "crop", "spotlight", "random_crop"])
    parser.add_argument("--vcd-noise-step", type=int, default=200)
    parser.add_argument("--vcd-noise-seed", type=int, default=42)
    parser.add_argument("--region-num", type=int, default=16)
    parser.add_argument("--region-quantile", type=float, default=0.5)
    parser.add_argument("--region-weight", type=float, default=2.5)
    parser.add_argument("--region-feather", type=float, default=0.02)
    parser.add_argument("--region-min-highlight", type=int, default=4)
    parser.add_argument("--crop-frac", type=float, default=0.6)
    parser.add_argument("--crop-margin", type=float, default=0.1)

    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-records", action="store_true")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def ensure_mmada_imports(mmada_root: str):
    mmada_path = Path(mmada_root).resolve()
    if not mmada_path.exists():
        raise FileNotFoundError(f"MMaDA root not found: {mmada_path}")
    path_str = str(mmada_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    from models import MAGVITv2, MMadaModelLM
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform, image_transform_squash

    return MAGVITv2, MMadaModelLM, UniversalPrompting, image_transform, image_transform_squash


class MMaDARunner:
    def __init__(self, args):
        (
            magvitv2_cls,
            mmada_model_cls,
            prompting_cls,
            image_transform_fn,
            image_transform_squash_fn,
        ) = ensure_mmada_imports(args.mmada_root)
        self.args = args
        self.device = resolve_device(args.device)
        self.dtype = get_torch_dtype(args.torch_dtype)
        self.image_transform = (
            image_transform_squash_fn if args.image_transform == "squash" else image_transform_fn
        )

        from transformers import AutoTokenizer

        tokenizer_path = args.tokenizer_path or args.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        self.uni_prompting = prompting_cls(
            self.tokenizer,
            max_text_len=args.max_text_len,
            special_tokens=SPECIAL_TOKENS,
            ignore_id=-100,
            cond_dropout_prob=0.0,
            use_reserved_token=True,
        )
        self.vq_model = magvitv2_cls.from_pretrained(args.vq_model_path).to(self.device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)

        self.model = mmada_model_cls.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.mask_token_id = (
            args.mask_token_id
            if args.mask_token_id is not None
            else getattr(self.model.config, "mask_token_id", DEFAULT_MASK_TOKEN_ID)
        )

    def image_to_codes(self, image):
        tensor = self.image_transform(
            image.convert("RGB"), resolution=self.args.resolution
        ).to(self.device)
        tensor = tensor.unsqueeze(0)
        return self.vq_model.get_code(tensor) + len(self.uni_prompting.text_tokenizer)

    def text_to_ids(self, prompt: str):
        if self.args.chat_template == "tokenizer" and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

        input_text = (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        ids = self.tokenizer([input_text], return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def build_input_ids(self, doc, prompt: str, image=None):
        image = image if image is not None else doc["image"]
        image_tokens = self.image_to_codes(image)
        text_token_ids = self.text_to_ids(prompt)
        batch_size = image_tokens.shape[0]
        task = torch.full(
            (batch_size, 1),
            int(self.uni_prompting.sptids_dict["<|mmu|>"]),
            device=self.device,
            dtype=torch.long,
        )
        soi = torch.full(
            (batch_size, 1),
            int(self.uni_prompting.sptids_dict["<|soi|>"]),
            device=self.device,
            dtype=torch.long,
        )
        eoi = torch.full(
            (batch_size, 1),
            int(self.uni_prompting.sptids_dict["<|eoi|>"]),
            device=self.device,
            dtype=torch.long,
        )
        return torch.cat([task, soi, image_tokens, eoi, text_token_ids], dim=1).long()

    def decode_answer_ids(self, answer_ids):
        if isinstance(answer_ids, torch.Tensor):
            answer_ids = answer_ids.detach().cpu().tolist()
        text = self.uni_prompting.text_tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return clean_generated_text(text).strip()

    @torch.no_grad()
    def baseline_generate(self, input_ids):
        kwargs = dict(
            max_new_tokens=self.args.max_new_tokens,
            steps=self.args.steps,
            block_length=self.args.block_length,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            cfg_scale=self.args.cfg_scale,
            remasking=self.args.remasking,
            mask_id=self.mask_token_id,
        )
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast("cuda", dtype=self.dtype):
                output_ids = self.model.mmu_generate(input_ids, **kwargs)
        else:
            output_ids = self.model.mmu_generate(input_ids, **kwargs)
        generated_ids = output_ids[:, input_ids.shape[1] :]
        return {
            "text": self.decode_answer_ids(generated_ids[0]),
            "answer_ids": generated_ids[0].detach().cpu().tolist(),
        }

    def compute_confidence(self, logits, x0):
        if self.args.remasking == "low_confidence":
            probs = F.softmax(logits.to(torch.float64), dim=-1)
            return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        if self.args.remasking == "random":
            return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        raise NotImplementedError(self.args.remasking)

    @staticmethod
    def get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = (
            torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
            + base
        )
        for idx in range(mask_num.size(0)):
            num_transfer_tokens[idx, : remainder[idx]] += 1
        return num_transfer_tokens

    def maybe_autocast(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    def forward_strict_mmada_logits(self, x, prompt_index, attention_bias=None):
        with self.maybe_autocast():
            if self.args.cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = self.mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = self.model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                return un_logits + (self.args.cfg_scale + 1) * (logits - un_logits)
            return self.model(x, attention_bias=attention_bias).logits

    def resolve_postvrg_steps(self):
        total_steps = int(self.args.steps)
        draft_steps = self.args.draft_steps
        postmask_steps = self.args.postmask_steps
        if draft_steps is None and postmask_steps is None:
            draft_steps = total_steps // 2
            postmask_steps = total_steps - draft_steps
        elif draft_steps is None:
            postmask_steps = int(postmask_steps)
            draft_steps = total_steps - postmask_steps
        elif postmask_steps is None:
            draft_steps = int(draft_steps)
            postmask_steps = total_steps - draft_steps
        else:
            draft_steps = int(draft_steps)
            postmask_steps = int(postmask_steps)
        if draft_steps <= 0:
            raise ValueError("draft_steps must be > 0.")
        if postmask_steps < 0:
            raise ValueError("postmask_steps must be >= 0.")
        if draft_steps + postmask_steps != total_steps:
            raise ValueError("draft_steps + postmask_steps must equal --steps.")
        return total_steps, draft_steps, postmask_steps

    def strict_mmada_block_denoise_stage(
        self,
        x,
        prompt_index,
        prefix_length,
        stage_steps,
        phase,
        global_step_offset=0,
        proposal_confidence=None,
        update_proposal_confidence=False,
        draft_answer_tokens=None,
        draft_answer_confidence=None,
        fixed_remask_positions=None,
        attention_bias=None,
    ):
        """MMaDA's official block-by-block denoising loop, with optional records.

        This is intentionally shaped like ``MMadaModelLM.mmu_generate``: generation
        is split into blocks, each block receives ``stage_steps / num_blocks``
        denoising iterations, future blocks are hidden from top-k transfer, and
        confidence/remasking are computed exactly from the current forward pass.
        """
        max_new_tokens = int(self.args.max_new_tokens)
        block_length = int(self.args.block_length)
        num_blocks = max_new_tokens // block_length
        if stage_steps == 0:
            return []
        if stage_steps % num_blocks != 0:
            raise ValueError(
                f"{phase} stage steps ({stage_steps}) must be divisible by num_blocks ({num_blocks})."
            )
        steps_per_block = stage_steps // num_blocks
        answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
        records = []
        step_counter = 0

        for num_block in range(num_blocks):
            block_start = prefix_length + num_block * block_length
            block_end = prefix_length + (num_block + 1) * block_length
            block_mask_index = x[:, block_start:block_end] == self.mask_token_id
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_in_block in range(steps_per_block):
                step_counter += 1
                mask_index = x == self.mask_token_id
                logits = self.forward_strict_mmada_logits(
                    x,
                    prompt_index=prompt_index,
                    attention_bias=attention_bias,
                )
                logits_with_noise = self.add_gumbel_noise(logits, self.args.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0_p = self.compute_confidence(logits, x0)

                # Official MMaDA suppresses future blocks before top-k transfer.
                # Suppressing earlier positions is a no-op for the official loop
                # because previous blocks are already filled, and it keeps fixed
                # PostVRG refill strictly inside the current block.
                x0_p[:, :block_start] = -torch.inf
                x0_p[:, block_end:] = -torch.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                scheduled_k = int(num_transfer_tokens[0, step_in_block].item())
                transfer_k = scheduled_k
                if phase == "refine" and self.args.fixed_refill_per_step is not None:
                    current_block_remaining = int((x[:, block_start:block_end] == self.mask_token_id).sum().item())
                    transfer_k = min(int(self.args.fixed_refill_per_step), current_block_remaining)

                selected = x.new_empty(0, dtype=torch.long)
                if transfer_k > 0:
                    _, selected = torch.topk(confidence[0], k=transfer_k)
                    answer_positions = selected - prefix_length
                    valid_answer_positions = (
                        (answer_positions >= 0) & (answer_positions < max_new_tokens)
                    )
                    if update_proposal_confidence and proposal_confidence is not None:
                        proposal_confidence[answer_positions[valid_answer_positions]] = (
                            confidence[0, selected[valid_answer_positions]]
                            .detach()
                            .to(torch.float64)
                        )

                    if (
                        phase == "refine"
                        and self.args.refill_confidence_gate
                        and draft_answer_tokens is not None
                        and draft_answer_confidence is not None
                    ):
                        ans_pos = answer_positions[valid_answer_positions]
                        selected_answer = selected[valid_answer_positions]
                        refill_conf = confidence[0, selected_answer]
                        draft_conf = draft_answer_confidence[ans_pos].to(refill_conf.dtype)
                        chosen = torch.where(
                            refill_conf >= draft_conf,
                            x0[0, selected_answer],
                            draft_answer_tokens[ans_pos],
                        )
                        x[0, selected_answer] = chosen
                    else:
                        x[0, selected] = x0[0, selected]

                state_ids = x[0, answer_slice].detach().cpu().tolist()
                selected_answer_positions = [
                    int(pos - prefix_length)
                    for pos in selected.detach().cpu().tolist()
                    if prefix_length <= pos < prefix_length + max_new_tokens
                ]
                record = {
                    "step": int(global_step_offset + step_counter),
                    "phase": phase,
                    "block": int(num_block),
                    "step_in_block": int(step_in_block + 1),
                    "scheduled_transfer_tokens": int(scheduled_k),
                    "num_filled": int(len(selected_answer_positions)),
                    "selected_answer_positions": selected_answer_positions,
                    "state_text": self.decode_answer_ids(state_ids),
                    "num_masked_after_step": int((x[:, answer_slice] == self.mask_token_id).sum().item()),
                }
                if phase == "refine":
                    if fixed_remask_positions is None:
                        remasked_positions = []
                    else:
                        remasked_positions = [
                            int(pos) for pos in fixed_remask_positions.detach().cpu().tolist()
                        ]
                    record["remasked_answer_positions"] = remasked_positions
                    record["refilled_answer_positions"] = selected_answer_positions
                records.append(record)

        return records

    @torch.no_grad()
    def postvrg_generate(self, draft_input_ids, refine_input_ids=None):
        total_steps, draft_steps, postmask_steps = self.resolve_postvrg_steps()
        max_new_tokens = int(self.args.max_new_tokens)

        if draft_input_ids.shape[0] != 1:
            raise ValueError("postvrg_generate expects batch size 1.")
        if max_new_tokens % self.args.block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length.")
        prefix_length = draft_input_ids.shape[1]
        num_blocks = max_new_tokens // int(self.args.block_length)
        if draft_steps % num_blocks != 0:
            raise ValueError("draft_steps must be divisible by the number of MMaDA generation blocks.")
        if postmask_steps > 0 and postmask_steps % num_blocks != 0:
            raise ValueError("postmask_steps must be divisible by the number of MMaDA generation blocks.")

        x = torch.full(
            (1, prefix_length + max_new_tokens),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prefix_length] = draft_input_ids.clone()
        prompt_index = x != self.mask_token_id
        answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
        proposal_confidence = torch.full(
            (max_new_tokens,), float("inf"), dtype=torch.float64, device=self.device
        )

        draft_records = self.strict_mmada_block_denoise_stage(
            x=x,
            prompt_index=prompt_index,
            prefix_length=prefix_length,
            stage_steps=draft_steps,
            phase="draft",
            global_step_offset=0,
            proposal_confidence=proposal_confidence,
            update_proposal_confidence=True,
        )

        draft_answer_ids = x[0, answer_slice].detach().cpu().tolist()
        draft_text = self.decode_answer_ids(draft_answer_ids)
        draft_answer_tokens = x[0, answer_slice].clone()
        draft_answer_confidence = proposal_confidence.clone()

        fixed_remask_positions = None
        if postmask_steps > 0:
            effective_fixed_set_size = min(
                int(self.args.fixed_set_size)
                if self.args.fixed_set_size is not None
                else max_new_tokens // 2,
                max_new_tokens,
            )
            fixed_remask_positions = torch.topk(
                proposal_confidence, k=effective_fixed_set_size, largest=False
            ).indices
            if fixed_remask_positions.numel() > 0:
                x[0, fixed_remask_positions + prefix_length] = self.mask_token_id

        if refine_input_ids is not None:
            if refine_input_ids.shape[1] != prefix_length:
                raise ValueError(
                    f"refine input length {refine_input_ids.shape[1]} != draft input length {prefix_length}."
                )
            x[:, :prefix_length] = refine_input_ids.clone()

        postmask_records = []
        if postmask_steps > 0 and fixed_remask_positions is not None and fixed_remask_positions.numel() > 0:
            postmask_records = self.strict_mmada_block_denoise_stage(
                x=x,
                prompt_index=prompt_index,
                prefix_length=prefix_length,
                stage_steps=postmask_steps,
                phase="refine",
                global_step_offset=draft_steps,
                draft_answer_tokens=draft_answer_tokens,
                draft_answer_confidence=draft_answer_confidence,
                fixed_remask_positions=fixed_remask_positions,
            )

        final_answer_ids = x[0, answer_slice].detach().cpu().tolist()
        final_text = self.decode_answer_ids(final_answer_ids)
        finite_confidence = proposal_confidence[torch.isfinite(proposal_confidence)]
        return {
            "draft_text": draft_text,
            "draft_answer_ids": draft_answer_ids,
            "draft_records": draft_records,
            "final_text": final_text,
            "final_answer_ids": final_answer_ids,
            "postmask_records": postmask_records,
            "meta": {
                "max_new_tokens": max_new_tokens,
                "block_length": int(self.args.block_length),
                "total_steps": int(total_steps),
                "draft_steps": int(draft_steps),
                "postmask_steps": int(postmask_steps),
                "num_blocks": int(num_blocks),
                "draft_steps_per_block": int(draft_steps // num_blocks),
                "postmask_steps_per_block": int(postmask_steps // num_blocks)
                if postmask_steps
                else 0,
                "draft_schedule": "mmada_block_transfer",
                "refine_schedule": "fixed_refill_per_step"
                if self.args.fixed_refill_per_step is not None
                else "mmada_block_transfer",
                "fixed_set_size": int(self.args.fixed_set_size)
                if self.args.fixed_set_size is not None
                else None,
                "fixed_refill_per_step": int(self.args.fixed_refill_per_step)
                if self.args.fixed_refill_per_step is not None
                else None,
                "proposal_confidence_mean": float(finite_confidence.mean().item())
                if finite_confidence.numel()
                else None,
            },
        }

    @staticmethod
    def add_gumbel_noise(logits, temperature):
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise


def select_edge_regions(args, image):
    if args.draft_visual_mode != "edge_noise" and args.refine_visual_mode not in ("crop", "spotlight"):
        return None
    from M3CoT.PostVRG.region_utils_final import image_to_gray_norm, select_highlight_regions

    return select_highlight_regions(
        image_to_gray_norm(image.convert("RGB")),
        gps_num=args.region_num,
        quantile=args.region_quantile,
        weight=args.region_weight,
        min_highlight=args.region_min_highlight,
    )


def build_draft_image(args, doc, edge_regions=None):
    image = doc["image"].convert("RGB")
    if args.draft_visual_mode == "full":
        return image
    if args.draft_visual_mode == "edge_noise":
        from M3CoT.PostVRG.region_utils_final import apply_region_corruption

        return apply_region_corruption(
            image,
            gps_num=args.region_num,
            quantile=args.region_quantile,
            weight=args.region_weight,
            feather_frac=args.region_feather,
            noise_step=args.vcd_noise_step,
            seed=args.vcd_noise_seed,
            min_highlight=args.region_min_highlight,
            regions=edge_regions,
        )
    if args.draft_visual_mode == "random_noise":
        from M3CoT.PostVRG.region_utils_final import apply_box_noise, random_box

        W, H = image.size
        box = random_box(doc.get("id", ""), W, H, args.crop_frac)
        return apply_box_noise(
            image,
            box,
            noise_step=args.vcd_noise_step,
            seed=args.vcd_noise_seed,
            feather_frac=args.region_feather,
        )
    raise ValueError(f"Unsupported draft visual mode: {args.draft_visual_mode}")


def build_refine_image(args, doc, edge_regions=None):
    image = doc["image"].convert("RGB")
    if args.refine_visual_mode == "full":
        return image
    if args.refine_visual_mode == "spotlight":
        from M3CoT.PostVRG.region_utils_final import apply_region_spotlight

        return apply_region_spotlight(
            image,
            gps_num=args.region_num,
            quantile=args.region_quantile,
            weight=args.region_weight,
            feather_frac=args.region_feather,
            noise_step=args.vcd_noise_step,
            seed=args.vcd_noise_seed,
            min_highlight=args.region_min_highlight,
            regions=edge_regions,
        )
    if args.refine_visual_mode == "crop":
        from M3CoT.PostVRG.region_utils_final import image_to_gray_norm, select_highlight_regions

        W, H = image.size
        if edge_regions is not None:
            regions_px, _, hi, _ = edge_regions
        else:
            regions_px, _, hi, _ = select_highlight_regions(
                image_to_gray_norm(image),
                gps_num=args.region_num,
                quantile=args.region_quantile,
                weight=args.region_weight,
                min_highlight=args.region_min_highlight,
            )
        hi_boxes = [regions_px[i] for i in range(len(regions_px)) if hi[i]]
        if not hi_boxes:
            hi_boxes = regions_px
        x1 = min(box[0] for box in hi_boxes)
        y1 = min(box[1] for box in hi_boxes)
        x2 = max(box[2] for box in hi_boxes)
        y2 = max(box[3] for box in hi_boxes)
        mx = int(args.crop_margin * (x2 - x1))
        my = int(args.crop_margin * (y2 - y1))
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(W - 1, x2 + mx)
        y2 = min(H - 1, y2 + my)
        return image.crop((x1, y1, x2 + 1, y2 + 1)).resize((W, H))
    if args.refine_visual_mode == "random_crop":
        from M3CoT.PostVRG.region_utils_final import random_box

        W, H = image.size
        x1, y1, x2, y2 = random_box(doc.get("id", ""), W, H, args.crop_frac)
        return image.crop((x1, y1, x2 + 1, y2 + 1)).resize((W, H))
    raise ValueError(f"Unsupported refine visual mode: {args.refine_visual_mode}")


def validate_generation_args(args):
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0.")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.block_length <= 0:
        raise ValueError("--block-length must be > 0.")
    if args.max_new_tokens % args.block_length != 0:
        raise ValueError("Strict MMaDA generation requires max_new_tokens % block_length == 0.")
    if args.block_length != args.max_new_tokens:
        raise ValueError("This runner is configured for a single block: block_length must equal max_new_tokens.")
    num_blocks = args.max_new_tokens // args.block_length
    if args.mode == "baseline":
        if args.steps % num_blocks != 0:
            raise ValueError("MMaDA mmu_generate requires steps % num_blocks == 0.")
    if args.mode == "postvrg":
        draft_steps = args.draft_steps
        postmask_steps = args.postmask_steps
        if draft_steps is None and postmask_steps is None:
            draft_steps = args.steps // 2
            postmask_steps = args.steps - draft_steps
        elif draft_steps is None:
            draft_steps = args.steps - int(postmask_steps)
        elif postmask_steps is None:
            postmask_steps = args.steps - int(draft_steps)
        draft_steps = int(draft_steps)
        postmask_steps = int(postmask_steps)
        if draft_steps + postmask_steps != args.steps:
            raise ValueError("draft_steps + postmask_steps must equal --steps.")
        if draft_steps % num_blocks != 0:
            raise ValueError("Strict MMaDA PostVRG requires draft_steps % num_blocks == 0.")
        if postmask_steps > 0 and postmask_steps % num_blocks != 0:
            raise ValueError("Strict MMaDA PostVRG requires postmask_steps % num_blocks == 0.")
        if args.refine_visual_mode != "full" and args.image_transform != "squash":
            print(
                "Warning: non-full refine images may change visual crop content under center_crop. "
                "Use --image-transform squash for strict whole-image resize behavior.",
                flush=True,
            )


def main():
    args = parse_args()
    validate_generation_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    runner = MMaDARunner(args)

    dataset = load_postvrg_dataset(args)
    if args.domain_filter:
        dataset = dataset.filter(lambda row: row.get("domain") == args.domain_filter)
    if args.sample_mode == "random":
        dataset = dataset.shuffle(seed=args.sample_seed)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    write_records = not args.no_records

    total_elapsed = 0.0
    written = 0
    correct_total = 0
    draft_correct_total = 0
    final_correct_total = 0
    improved_total = 0
    worsened_total = 0
    failures = 0

    with (records_path.open("w", encoding="utf-8") if write_records else contextlib.nullcontext()) as fout:
        for dataset_index, doc in enumerate(dataset):
            if doc.get("image") is None:
                continue
            t0 = time.time()
            context = build_prompt(doc, args.prompt)
            prompt = context
            try:
                if args.mode == "baseline":
                    input_ids = runner.build_input_ids(doc, prompt)
                    run_output = runner.baseline_generate(input_ids)
                    elapsed = time.time() - t0
                    total_elapsed += elapsed
                    correct = bool(judge_answer(run_output["text"], doc["choices"], doc["answer"]))
                    correct_total += int(correct)
                    record = {
                        "dataset_index": int(dataset_index),
                        "id": doc["id"],
                        "question": context,
                        "choices": list(doc["choices"]),
                        "answer": doc["answer"],
                        "domain": doc["domain"],
                        "topic": doc["topic"],
                        "prompt": prompt,
                        "elapsed_sec": elapsed,
                        "response_text": run_output["text"],
                        "answer_ids": run_output["answer_ids"],
                        "correct": correct,
                        "meta": {
                            "mode": "baseline",
                            "max_new_tokens": args.max_new_tokens,
                            "steps": args.steps,
                            "block_length": args.block_length,
                        },
                    }
                else:
                    edge_regions = select_edge_regions(args, doc["image"])
                    draft_image = build_draft_image(args, doc, edge_regions=edge_regions)
                    refine_image = build_refine_image(args, doc, edge_regions=edge_regions)
                    draft_input_ids = runner.build_input_ids(doc, prompt, image=draft_image)
                    refine_input_ids = None
                    if args.refine_visual_mode != args.draft_visual_mode:
                        refine_input_ids = runner.build_input_ids(doc, prompt, image=refine_image)
                    run_output = runner.postvrg_generate(draft_input_ids, refine_input_ids=refine_input_ids)
                    elapsed = time.time() - t0
                    total_elapsed += elapsed
                    draft_correct = bool(judge_answer(run_output["draft_text"], doc["choices"], doc["answer"]))
                    final_correct = bool(judge_answer(run_output["final_text"], doc["choices"], doc["answer"]))
                    draft_correct_total += int(draft_correct)
                    final_correct_total += int(final_correct)
                    if final_correct and not draft_correct:
                        improved_total += 1
                    if draft_correct and not final_correct:
                        worsened_total += 1
                    record = {
                        "dataset_index": int(dataset_index),
                        "id": doc["id"],
                        "question": context,
                        "choices": list(doc["choices"]),
                        "answer": doc["answer"],
                        "domain": doc["domain"],
                        "topic": doc["topic"],
                        "prompt": prompt,
                        "elapsed_sec": elapsed,
                        "draft_text": run_output["draft_text"],
                        "draft_answer_ids": run_output["draft_answer_ids"],
                        "draft_correct": draft_correct,
                        "final_text": run_output["final_text"],
                        "final_answer_ids": run_output["final_answer_ids"],
                        "final_correct": final_correct,
                        "draft_records": run_output["draft_records"],
                        "postmask_records": run_output["postmask_records"],
                        "meta": run_output["meta"],
                    }
            except Exception as exc:
                failures += 1
                elapsed = time.time() - t0
                total_elapsed += elapsed
                record = {
                    "dataset_index": int(dataset_index),
                    "id": doc.get("id"),
                    "question": context,
                    "choices": list(doc.get("choices", [])),
                    "answer": doc.get("answer"),
                    "domain": doc.get("domain"),
                    "topic": doc.get("topic"),
                    "prompt": prompt,
                    "elapsed_sec": elapsed,
                    "error": repr(exc),
                    "correct": False,
                }
                print(f"[Error] dataset_index={dataset_index} id={doc.get('id')}: {exc}", flush=True)

            if write_records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                if args.mode == "baseline":
                    status = f"correct={record.get('correct')}"
                else:
                    status = f"draft={record.get('draft_correct')} final={record.get('final_correct')}"
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc.get('id')} "
                    f"{status} elapsed={record.get('elapsed_sec', 0.0):.2f}s",
                    flush=True,
                )

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "benchmark": args.benchmark,
        "start_index": args.start_index,
        "domain_filter": args.domain_filter,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed if args.sample_mode == "random" else None,
        "num_samples": written,
        "failures": failures,
        "prompt": args.prompt,
        "mode": args.mode,
        "model_path": args.model_path,
        "vq_model_path": args.vq_model_path,
        "mmada_root": args.mmada_root,
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "mask_token_id": runner.mask_token_id,
            "resolution": args.resolution,
            "image_transform": args.image_transform,
            "chat_template": args.chat_template,
        },
    }
    if args.mode == "baseline":
        summary["accuracy"] = correct_total / written if written else None
    else:
        summary.update(
            {
                "draft_accuracy": draft_correct_total / written if written else None,
                "final_accuracy": final_correct_total / written if written else None,
                "improved_after_postmask": int(improved_total),
                "worsened_after_postmask": int(worsened_total),
            }
        )
        summary["generation"].update(
            {
                "draft_steps": args.draft_steps,
                "postmask_steps": args.postmask_steps,
                "fixed_set_size": args.fixed_set_size,
                "fixed_refill_per_step": args.fixed_refill_per_step,
                "refill_confidence_gate": args.refill_confidence_gate,
                "draft_visual_mode": args.draft_visual_mode,
                "refine_visual_mode": args.refine_visual_mode,
                "vcd_noise_step": args.vcd_noise_step
                if args.draft_visual_mode != "full" or args.refine_visual_mode == "spotlight"
                else None,
                "vcd_noise_seed": args.vcd_noise_seed
                if args.draft_visual_mode != "full" or args.refine_visual_mode == "spotlight"
                else None,
                "crop_margin": args.crop_margin if args.refine_visual_mode == "crop" else None,
                "crop_frac": args.crop_frac
                if args.refine_visual_mode == "random_crop" or args.draft_visual_mode == "random_noise"
                else None,
            }
        )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    apply_mmada_defaults()
    main()
