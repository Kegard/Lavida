from __future__ import annotations

import logging
import math
import re
from contextlib import contextmanager
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


eval_logger = logging.getLogger("lmms-eval")


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _as_int(x: Any, default: int) -> int:
    if x is None:
        return int(default)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return int(default)
        return int(s)
    return int(x)


def infer_multimodal_layout(
    model_obj,
    input_ids: torch.Tensor,
    images,
    image_sizes,
) -> Dict[str, Any]:
    attention_mask = input_ids.new_ones(input_ids.shape, dtype=input_ids.dtype)
    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model_obj,
        input_ids=input_ids,
        images=images,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
    )
    del prefix_embeds

    valid_ids = prefix_input_ids_full[prefix_input_ids_full != IGNORE_INDEX]
    visual_positions = (valid_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if visual_positions.numel() == 0:
        raise RuntimeError("No visual token positions found in the expanded multimodal prefix.")

    visual_position_list = [int(idx) for idx in visual_positions.tolist()]
    try:
        vision_tower = model_obj.get_vision_tower()
        if hasattr(vision_tower, "num_patches"):
            expected_patches = int(vision_tower.num_patches)
        elif hasattr(vision_tower, "num_patches_per_side"):
            patches_per_side = int(vision_tower.num_patches_per_side)
            expected_patches = patches_per_side * patches_per_side
        else:
            expected_patches = 0
        if (
            expected_patches > 0
            and len(visual_position_list) == expected_patches + 1
            and visual_position_list == list(range(visual_position_list[0], visual_position_list[0] + len(visual_position_list)))
        ):
            visual_position_list = visual_position_list[:expected_patches]
    except Exception:
        pass

    text_query_positions = [int(i) for i, token_id in enumerate(valid_ids.tolist()) if token_id != IMAGE_TOKEN_INDEX]
    return {
        "visual_positions": visual_position_list,
        "text_query_positions": text_query_positions,
        "prefix_length": int(valid_ids.shape[0]),
    }


def resolve_layer_indices(num_layers: int, spec: str) -> List[int]:
    value = (spec or "last").strip().lower()
    if value == "all":
        return list(range(num_layers))
    if value == "last":
        return [num_layers - 1]
    if re.fullmatch(r"last\d+", value):
        count = max(1, min(num_layers, int(value[4:])))
        return list(range(num_layers - count, num_layers))

    parts = [part.strip() for part in re.split(r"[|:]", value) if part.strip()]
    if not parts:
        raise ValueError(f"Invalid sink_layers spec: {spec}")

    resolved: List[int] = []
    for part in parts:
        idx = int(part)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index out of range in sink_layers: {part}")
        resolved.append(idx)
    return sorted(set(resolved))


def parse_step_modes(spec: str) -> List[str]:
    value = (spec or "both").strip().lower()
    if value == "both":
        return ["prefill", "decode"]
    if value in {"prefill", "decode"}:
        return [value]
    raise ValueError(f"Unsupported sink_steps value: {spec}")


def build_sink_config(**kwargs) -> Dict[str, Any]:
    return {
        "enabled": _as_bool(kwargs.get("enabled", False)),
        "intervention": str(kwargs.get("intervention", "none")).strip().lower(),
        "selector": str(kwargs.get("selector", "top_attn")).strip().lower(),
        "topk": max(1, _as_int(kwargs.get("topk", 1), 1)),
        "query_scope": str(kwargs.get("query_scope", "text")).strip().lower(),
        "control": str(kwargs.get("control", "none")).strip().lower(),
        "head_scope": str(kwargs.get("head_scope", "all")).strip().lower(),
        "seed": _as_int(kwargs.get("seed", 0), 0),
        "debug": _as_bool(kwargs.get("debug", False)),
        "step_modes": parse_step_modes(str(kwargs.get("steps", "both"))),
    }


def _parse_intervention_spec(spec: str) -> Tuple[str, float]:
    value = (spec or "none").strip().lower()
    soft_debias_match = re.fullmatch(r"positive_cos_soft_debias_(\d+(?:p\d+)?)(?:_t\d+(?:p\d+)?)?", value)
    if soft_debias_match:
        return "positive_cos_soft_debias", float(soft_debias_match.group(1).replace("p", "."))
    outro_match = re.fullmatch(r"outro_rotate_(\d+(?:p\d+)?)(?:_t\d+(?:p\d+)?)?", value)
    if outro_match:
        return "outro_rotate", float(outro_match.group(1).replace("p", "."))
    match = re.fullmatch(r"(direction_debias|direction_logit_penalty|partial_zero_v|logit_penalty)_(\d+(?:p\d+)?)", value)
    if match:
        return match.group(1), float(match.group(2).replace("p", "."))
    if value == "positive_cos_soft_debias":
        return value, 1.0
    if value == "outro_rotate":
        return value, 1.0
    if value == "direction_debias":
        return value, 0.5
    if value == "direction_logit_penalty":
        return value, 1.0
    if value == "partial_zero_v":
        return value, 0.5
    if value == "logit_penalty":
        return value, 2.0
    return value, 0.0


def _parse_soft_debias_threshold(spec: str) -> float:
    value = (spec or "").strip().lower()
    match = re.fullmatch(r"positive_cos_soft_debias_\d+(?:p\d+)?(?:_t(\d+(?:p\d+)?))?", value)
    if match and match.group(1) is not None:
        return float(match.group(1).replace("p", "."))
    return 0.0


def _parse_outro_temperature(spec: str) -> float:
    value = (spec or "").strip().lower()
    match = re.fullmatch(r"outro_rotate_\d+(?:p\d+)?(?:_t(\d+(?:p\d+)?))?", value)
    if match and match.group(1) is not None:
        return float(match.group(1).replace("p", "."))
    return 0.1


def _select_query_positions(
    *,
    current_stage: str,
    query_scope: str,
    query_len: int,
    text_query_positions: Sequence[int],
) -> List[int]:
    if query_len <= 0:
        return []
    if current_stage == "prefill":
        if query_scope == "text":
            positions = [int(idx) for idx in text_query_positions if 0 <= int(idx) < query_len]
            if positions:
                return positions
        return list(range(query_len))
    return list(range(query_len))


def _repeat_kv_for_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.shape[1] == k.shape[1]:
        return q, k, v
    repeat = q.shape[1] // k.shape[1]
    return q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def _compute_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_bias: Optional[torch.Tensor],
    block_module,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attention_bias is not None:
        query_len, key_len = scores.shape[-2], scores.shape[-1]
        bias = block_module._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len],
            scores.dtype,
        )
        scores = scores + bias
    return scores


def _rank_positions(desc_values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(desc_values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return ranks


def _resolve_control_positions(
    *,
    base_indices: torch.Tensor,
    attn_mean: torch.Tensor,
    key_norm_mean: torch.Tensor,
    control: str,
    topk: int,
    seed: int,
) -> torch.Tensor:
    num_visual = int(attn_mean.numel())
    if num_visual <= 0 or control == "none":
        return base_indices

    device = attn_mean.device
    all_indices = torch.arange(num_visual, device=device)
    base_mask = torch.zeros(num_visual, dtype=torch.bool, device=device)
    base_mask[base_indices] = True
    candidates = all_indices[~base_mask]
    if candidates.numel() == 0:
        return base_indices

    select_k = min(topk, int(candidates.numel()))
    if control == "random_visual":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        perm = torch.randperm(int(candidates.numel()), generator=generator, device=device)
        return candidates[perm[:select_k]]

    if control == "high_attention_non_sink":
        order = candidates[torch.argsort(attn_mean[candidates], descending=True)]
        return order[:select_k]

    if control == "matched_norm":
        target = key_norm_mean[base_indices].mean()
        order = candidates[torch.argsort(torch.abs(key_norm_mean[candidates] - target), descending=False)]
        return order[:select_k]

    raise ValueError(f"Unsupported sink_control value: {control}")


def _pick_visual_indices(
    *,
    selector: str,
    topk: int,
    attn_mean: torch.Tensor,
    cos_mean: torch.Tensor,
) -> torch.Tensor:
    select_k = min(topk, int(attn_mean.numel()))
    direction_score = attn_mean * torch.clamp(cos_mean, min=0.0)
    direction_mass_match = re.fullmatch(r"direction_sink_mass_(\d+(?:p\d+)?)", selector)
    if direction_mass_match:
        mass_ratio = float(direction_mass_match.group(1).replace("p", "."))
        mass_ratio = max(0.0, min(1.0, mass_ratio))
        order = torch.argsort(direction_score, descending=True)
        sorted_score = direction_score.index_select(0, order)
        total_score = sorted_score.sum()
        if total_score > 0:
            cumulative = torch.cumsum(sorted_score, dim=0) / total_score.clamp_min(1e-12)
            count = int(torch.searchsorted(cumulative, torch.tensor(mass_ratio, device=cumulative.device), right=False).item()) + 1
            return order[: max(1, min(count, int(order.numel())))]
        return torch.argsort(attn_mean, descending=True)[:1]
    if selector == "direction_sink":
        order = torch.argsort(direction_score, descending=True)
        if direction_score.index_select(0, order[:1]).item() > 0:
            return order[:select_k]
        return torch.argsort(attn_mean, descending=True)[:select_k]
    positive_cos_match = re.fullmatch(r"positive_cos(?:_(\d+(?:p\d+)?))?", selector)
    if positive_cos_match:
        threshold = 0.0
        if positive_cos_match.group(1) is not None:
            threshold = float(positive_cos_match.group(1).replace("p", "."))
        selected = torch.where(cos_mean > threshold)[0]
        if selected.numel() > 0:
            return selected
        return torch.argsort(cos_mean, descending=True)[:1]
    if selector == "top_attn":
        return torch.argsort(attn_mean, descending=True)[:select_k]
    if selector == "top_cos":
        return torch.argsort(cos_mean, descending=True)[:select_k]
    if selector == "top_attn_cos_intersection":
        attn_order = torch.argsort(attn_mean, descending=True)
        cos_order = torch.argsort(cos_mean, descending=True)
        pool = min(int(attn_mean.numel()), max(select_k, select_k * 4))
        cos_pool = set(int(x) for x in cos_order[:pool].tolist())
        picked = [int(idx) for idx in attn_order.tolist() if int(idx) in cos_pool][:select_k]
        if len(picked) >= select_k:
            return torch.tensor(picked, dtype=torch.long, device=attn_mean.device)
        attn_ranks = _rank_positions(attn_mean)
        cos_ranks = _rank_positions(cos_mean)
        score = attn_ranks + cos_ranks
        return torch.argsort(score, descending=False)[:select_k]
    raise ValueError(f"Unsupported sink_selector value: {selector}")


def _select_sink_positions(
    *,
    attn_scores: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    visual_positions: Sequence[int],
    query_positions: Sequence[int],
    selector: str,
    topk: int,
    control: str,
    head_scope: str,
    seed: int,
) -> Dict[str, Any]:
    if not visual_positions:
        return {
            "visual_local_indices": torch.empty((0,), dtype=torch.long, device=q.device),
            "selected_key_positions": [],
            "attn_mean": torch.empty((0,), dtype=torch.float32),
            "cos_mean": torch.empty((0,), dtype=torch.float32),
            "key_norm_mean": torch.empty((0,), dtype=torch.float32),
            "head_mask": None,
        }

    query_idx = torch.as_tensor(list(query_positions), dtype=torch.long, device=q.device)
    visual_idx = torch.as_tensor(list(visual_positions), dtype=torch.long, device=q.device)

    visual_scores = attn_scores.index_select(2, query_idx).index_select(3, visual_idx)
    visual_attn = F.softmax(visual_scores.to(torch.float32), dim=-1)
    attn_mean = visual_attn.mean(dim=(0, 1, 2))

    q_sel = q.index_select(2, query_idx)
    k_sel = k.index_select(2, visual_idx)
    q_unit = F.normalize(q_sel.to(torch.float32), p=2, dim=-1, eps=1e-12)
    k_unit = F.normalize(k_sel.to(torch.float32), p=2, dim=-1, eps=1e-12)
    cos_mean = (q_unit.unsqueeze(-2) * k_unit.unsqueeze(-3)).sum(dim=-1).mean(dim=(0, 1, 2))
    key_norm_mean = k_sel.to(torch.float32).norm(dim=-1).mean(dim=(0, 1))
    direction_score = attn_mean * torch.clamp(cos_mean, min=0.0)

    base_indices = _pick_visual_indices(
        selector=selector,
        topk=topk,
        attn_mean=attn_mean,
        cos_mean=cos_mean,
    )
    final_indices = _resolve_control_positions(
        base_indices=base_indices,
        attn_mean=attn_mean,
        key_norm_mean=key_norm_mean,
        control=control,
        topk=topk,
        seed=seed,
    )
    actual_positions = visual_idx.index_select(0, final_indices)
    head_mask = None
    if head_scope != "all":
        selected_idx = actual_positions
        full_attn = F.softmax(attn_scores.index_select(2, query_idx).to(torch.float32), dim=-1)
        visual_mass = full_attn.index_select(3, visual_idx).sum(dim=-1).mean(dim=(0, 2))
        sink_mass = full_attn.index_select(3, selected_idx).sum(dim=-1).mean(dim=(0, 2))
        if head_scope == "image":
            head_score = visual_mass
        elif head_scope == "sink":
            head_score = sink_mass
        elif head_scope == "image_sink":
            head_score = visual_mass * sink_mass
        else:
            raise ValueError(f"Unsupported sink_head_scope value: {head_scope}")
        threshold = head_score.mean()
        head_mask = head_score >= threshold
        if not bool(head_mask.any()):
            head_mask = None
    return {
        "visual_local_indices": final_indices,
        "selected_key_positions": [int(x) for x in actual_positions.tolist()],
        "attn_mean": attn_mean.detach().cpu(),
        "cos_mean": cos_mean.detach().cpu(),
        "key_norm_mean": key_norm_mean.detach().cpu(),
        "selected_direction_score": direction_score.index_select(0, final_indices).detach(),
        "base_visual_local_indices": base_indices.detach().cpu(),
        "head_mask": head_mask,
    }


def _apply_sink_intervention(
    *,
    intervention: str,
    attn_scores: torch.Tensor,
    k_for_cache: torch.Tensor,
    v_for_cache: torch.Tensor,
    selected_key_positions: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intervention_name, intervention_value = _parse_intervention_spec(intervention)
    if intervention_name in {"none", "direction_debias", "direction_logit_penalty", "outro_rotate"} or not selected_key_positions:
        return attn_scores, k_for_cache, v_for_cache

    selected = torch.as_tensor(list(selected_key_positions), dtype=torch.long, device=attn_scores.device)
    updated_scores = attn_scores
    updated_k = k_for_cache
    updated_v = v_for_cache

    if intervention_name in {"mask_attn", "remove"}:
        updated_scores = updated_scores.clone()
        updated_scores.index_fill_(-1, selected, torch.finfo(updated_scores.dtype).min)

    if intervention_name == "logit_penalty":
        penalty = float(intervention_value)
        updated_scores = updated_scores.clone()
        selected_scores = updated_scores.index_select(-1, selected) - penalty
        updated_scores.index_copy_(-1, selected, selected_scores)

    if intervention_name in {"zero_k", "remove"}:
        updated_k = updated_k.clone()
        updated_k.index_fill_(-2, selected, 0.0)

    if intervention_name in {"zero_v", "remove"}:
        updated_v = updated_v.clone()
        updated_v.index_fill_(-2, selected, 0.0)

    if intervention_name == "partial_zero_v":
        scale = float(intervention_value)
        updated_v = updated_v.clone()
        selected_v = updated_v.index_select(-2, selected) * scale
        updated_v.index_copy_(-2, selected, selected_v)

    return updated_scores, updated_k, updated_v


def _apply_direction_debias(
    *,
    q_for_attn: torch.Tensor,
    k_for_attn: torch.Tensor,
    selected_key_positions: Sequence[int],
    alpha: float,
) -> torch.Tensor:
    if not selected_key_positions:
        return k_for_attn

    selected = torch.as_tensor(list(selected_key_positions), dtype=torch.long, device=k_for_attn.device)
    mean_q = q_for_attn.to(torch.float32).mean(dim=-2, keepdim=True)
    mean_q = F.normalize(mean_q, p=2, dim=-1, eps=1e-12).to(dtype=k_for_attn.dtype)

    selected_k = k_for_attn.index_select(-2, selected)
    projection = (selected_k * mean_q).sum(dim=-1, keepdim=True) * mean_q
    debiased_k = selected_k - float(alpha) * projection

    updated_k = k_for_attn.clone()
    updated_k.index_copy_(-2, selected, debiased_k)
    return updated_k


def _apply_direction_logit_penalty(
    *,
    attn_scores: torch.Tensor,
    selected_key_positions: Sequence[int],
    selected_direction_score: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if not selected_key_positions:
        return attn_scores

    selected = torch.as_tensor(list(selected_key_positions), dtype=torch.long, device=attn_scores.device)
    weights = selected_direction_score.to(device=attn_scores.device, dtype=torch.float32)
    if weights.numel() != selected.numel():
        weights = torch.ones_like(selected, dtype=torch.float32, device=attn_scores.device)
    else:
        weights = weights / weights.max().clamp_min(1e-12)
    weights = weights.to(dtype=attn_scores.dtype)

    updated_scores = attn_scores.clone()
    selected_scores = updated_scores.index_select(-1, selected) - float(gamma) * weights.view(1, 1, 1, -1)
    updated_scores.index_copy_(-1, selected, selected_scores)
    return updated_scores


def _apply_positive_cos_soft_debias(
    *,
    q_for_attn: torch.Tensor,
    k_for_attn: torch.Tensor,
    visual_positions: Sequence[int],
    alpha: float,
    threshold: float,
) -> torch.Tensor:
    if not visual_positions:
        return k_for_attn

    visual_idx = torch.as_tensor(list(visual_positions), dtype=torch.long, device=k_for_attn.device)
    mean_q = q_for_attn.to(torch.float32).mean(dim=-2, keepdim=True)
    mean_q = F.normalize(mean_q, p=2, dim=-1, eps=1e-12)

    k_vis = k_for_attn.index_select(-2, visual_idx)
    k_vis_unit = F.normalize(k_vis.to(torch.float32), p=2, dim=-1, eps=1e-12)
    cos = (k_vis_unit * mean_q).sum(dim=-1, keepdim=True)

    weights = torch.clamp(cos - float(threshold), min=0.0)
    max_weight = weights.max()
    if max_weight <= 0:
        return k_for_attn
    weights = weights / max_weight.clamp_min(1e-12)

    mean_q = mean_q.to(dtype=k_for_attn.dtype)
    projection = (k_vis * mean_q).sum(dim=-1, keepdim=True) * mean_q
    debiased_k = k_vis - float(alpha) * weights.to(dtype=k_vis.dtype) * projection

    updated_k = k_for_attn.clone()
    updated_k.index_copy_(-2, visual_idx, debiased_k)
    return updated_k


def _apply_outro_rotate(
    *,
    attn_output: torch.Tensor,
    v_for_attn: torch.Tensor,
    selected_key_positions: Sequence[int],
    query_len: int,
    gamma: float,
    temperature: float,
    head_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not selected_key_positions:
        return attn_output

    selected = torch.as_tensor(list(selected_key_positions), dtype=torch.long, device=attn_output.device)
    selected = selected[(selected >= 0) & (selected < v_for_attn.shape[-2])]
    if selected.numel() == 0:
        return attn_output

    sink_v = v_for_attn.index_select(-2, selected).mean(dim=-2, keepdim=True)
    sink_v_float = sink_v.to(torch.float32)
    output_float = attn_output.to(torch.float32)

    sink_norm = sink_v_float.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    output_norm = output_float.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos = (output_float * sink_v_float).sum(dim=-1, keepdim=True) / (output_norm * sink_norm)
    gate = torch.tanh(torch.clamp(cos, min=0.0) / max(float(temperature), 1e-6))

    denom = (sink_v_float * sink_v_float).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection_scale = (output_float * sink_v_float).sum(dim=-1, keepdim=True) / denom
    rotated = output_float + float(gamma) * gate * projection_scale * sink_v_float

    rotated_norm = rotated.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rotated = output_norm * rotated / rotated_norm

    if head_mask is not None:
        head_mask = head_mask.to(device=attn_output.device, dtype=torch.bool)
        if head_mask.numel() == attn_output.shape[1] and bool(head_mask.any()):
            keep = ~head_mask.view(1, -1, 1, 1)
            rotated = torch.where(keep, output_float, rotated)

    if query_len == v_for_attn.shape[-2]:
        query_selected = selected[selected < query_len]
        if query_selected.numel() > 0:
            rotated = rotated.clone()
            rotated.index_copy_(-2, query_selected, output_float.index_select(-2, query_selected))

    return rotated.to(dtype=attn_output.dtype)


@contextmanager
def patch_attention_sink(
    *,
    model,
    layers_to_patch: List[int],
    visual_positions: Sequence[int],
    text_query_positions: Sequence[int],
    sink_config: Dict[str, Any],
):
    blocks = model.get_model().transformer.blocks
    original_methods: Dict[int, Any] = {}
    layer_state: Dict[str, Dict[str, Any]] = {}

    def make_patched_attention(block_module, layer_name_local: str, original_attention):
        def patched_attention(self_block, q, k, v, attention_bias=None, layer_past=None, use_cache=False, block_mask=None):
            current_stage = "decode" if layer_past is not None else "prefill"
            if current_stage not in sink_config["step_modes"]:
                return original_attention(
                    q,
                    k,
                    v,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    block_mask=block_mask,
                )

            B, T, C = q.size()
            dtype = k.dtype
            if block_module.q_norm is not None and block_module.k_norm is not None:
                q = block_module.q_norm(q).to(dtype=dtype)
                k = block_module.k_norm(k).to(dtype=dtype)

            q = q.view(B, T, block_module.config.n_heads, C // block_module.config.n_heads).transpose(1, 2)
            k = k.view(B, T, block_module.config.effective_n_kv_heads, C // block_module.config.n_heads).transpose(1, 2)
            v = v.view(B, T, block_module.config.effective_n_kv_heads, C // block_module.config.n_heads).transpose(1, 2)

            if layer_past is not None:
                past_key, past_value = layer_past
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)

            current_layer_state = layer_state.setdefault(
                layer_name_local,
                {
                    "selected_key_positions": None,
                    "selected_stage": None,
                },
            )

            q_for_attn = q
            k_for_cache = k
            v_for_cache = v
            if block_module.config.rope:
                q_for_attn, k_for_attn = block_module.rotary_emb(q_for_attn, k_for_cache)
            else:
                k_for_attn = k_for_cache

            q_for_attn, k_for_attn, v_for_attn = _repeat_kv_for_attention(q_for_attn, k_for_attn, v_for_cache)
            attn_scores = _compute_attention_scores(q_for_attn, k_for_attn, attention_bias, block_module)

            intervention_name, intervention_value = _parse_intervention_spec(sink_config["intervention"])
            dynamic_selection = intervention_name == "direction_logit_penalty"
            selected_direction_score = torch.empty((0,), dtype=torch.float32, device=attn_scores.device)

            if intervention_name == "positive_cos_soft_debias":
                threshold = _parse_soft_debias_threshold(sink_config["intervention"])
                modified_k_for_attn = _apply_positive_cos_soft_debias(
                    q_for_attn=q_for_attn,
                    k_for_attn=k_for_attn,
                    visual_positions=visual_positions,
                    alpha=float(intervention_value),
                    threshold=threshold,
                )
                modified_scores = _compute_attention_scores(q_for_attn, modified_k_for_attn, attention_bias, block_module)
                present = (k_for_cache, v_for_cache) if use_cache else None
                attn_weights = F.softmax(modified_scores, dim=-1).to(v_for_attn.dtype)
                att = torch.matmul(attn_weights, v_for_attn)
                att = att.transpose(1, 2).contiguous().view(B, T, C)
                return block_module.attn_out(att), present

            if dynamic_selection or current_layer_state["selected_key_positions"] is None:
                query_positions = _select_query_positions(
                    current_stage=current_stage,
                    query_scope=sink_config["query_scope"],
                    query_len=int(q_for_attn.shape[-2]),
                    text_query_positions=text_query_positions,
                )
                if query_positions and visual_positions:
                    selection = _select_sink_positions(
                        attn_scores=attn_scores,
                        q=q_for_attn,
                        k=k_for_attn,
                        visual_positions=visual_positions,
                        query_positions=query_positions,
                        selector=sink_config["selector"],
                        topk=sink_config["topk"],
                        control=sink_config["control"],
                        head_scope=sink_config["head_scope"],
                        seed=int(sink_config["seed"]) + int(re.sub(r"^\D+", "", layer_name_local) or 0),
                    )
                    current_layer_state["selected_key_positions"] = selection["selected_key_positions"]
                    current_layer_state["selected_stage"] = current_stage
                    current_layer_state["attn_mean"] = selection["attn_mean"]
                    current_layer_state["cos_mean"] = selection["cos_mean"]
                    current_layer_state["key_norm_mean"] = selection["key_norm_mean"]
                    current_layer_state["base_visual_local_indices"] = selection["base_visual_local_indices"]
                    current_layer_state["selected_direction_score"] = selection["selected_direction_score"]
                    current_layer_state["head_mask"] = selection["head_mask"]
                    if sink_config["debug"]:
                        eval_logger.warning(
                            "Sink selection %s stage=%s positions=%s",
                            layer_name_local,
                            current_stage,
                            current_layer_state["selected_key_positions"],
                        )

            selected_key_positions = current_layer_state.get("selected_key_positions") or []
            if "selected_direction_score" in current_layer_state:
                selected_direction_score = current_layer_state["selected_direction_score"]
            head_mask = current_layer_state.get("head_mask")
            if intervention_name == "none" or not selected_key_positions:
                present = (k_for_cache, v_for_cache) if use_cache else None
                attn_weights = F.softmax(attn_scores, dim=-1).to(v_for_attn.dtype)
                att = torch.matmul(attn_weights, v_for_attn)
                att = att.transpose(1, 2).contiguous().view(B, T, C)
                return block_module.attn_out(att), present

            raw_selected = torch.as_tensor(selected_key_positions, dtype=torch.long, device=k_for_cache.device)
            if q_for_attn.shape[1] != k_for_cache.shape[1]:
                repeat = q_for_attn.shape[1] // k_for_cache.shape[1]
                expanded_selected = raw_selected
            else:
                repeat = 1
                expanded_selected = raw_selected
            del repeat

            modified_scores, modified_k_cache, modified_v_cache = _apply_sink_intervention(
                intervention=sink_config["intervention"],
                attn_scores=attn_scores,
                k_for_cache=k_for_cache,
                v_for_cache=v_for_cache,
                selected_key_positions=expanded_selected.tolist(),
            )

            if block_module.config.rope:
                _, modified_k_for_attn = block_module.rotary_emb(q, modified_k_cache)
            else:
                modified_k_for_attn = modified_k_cache
            _, modified_k_for_attn, modified_v_for_attn = _repeat_kv_for_attention(q_for_attn, modified_k_for_attn, modified_v_cache)

            if intervention_name == "direction_debias":
                modified_k_for_attn = _apply_direction_debias(
                    q_for_attn=q_for_attn,
                    k_for_attn=modified_k_for_attn,
                    selected_key_positions=selected_key_positions,
                    alpha=float(intervention_value),
                )
                modified_scores = _compute_attention_scores(q_for_attn, modified_k_for_attn, attention_bias, block_module)

            if intervention_name == "direction_logit_penalty":
                modified_scores = _apply_direction_logit_penalty(
                    attn_scores=attn_scores,
                    selected_key_positions=selected_key_positions,
                    selected_direction_score=selected_direction_score,
                    gamma=float(intervention_value),
                )

            if intervention_name in {"zero_k", "zero_v", "partial_zero_v", "remove"}:
                modified_scores = _compute_attention_scores(q_for_attn, modified_k_for_attn, attention_bias, block_module)
                if intervention_name == "remove":
                    selected = torch.as_tensor(selected_key_positions, dtype=torch.long, device=modified_scores.device)
                    modified_scores = modified_scores.clone()
                    modified_scores.index_fill_(-1, selected, torch.finfo(modified_scores.dtype).min)

            present = (modified_k_cache, modified_v_cache) if use_cache else None
            attn_weights = F.softmax(modified_scores, dim=-1).to(modified_v_for_attn.dtype)
            att = torch.matmul(attn_weights, modified_v_for_attn)
            if intervention_name == "outro_rotate":
                att = _apply_outro_rotate(
                    attn_output=att,
                    v_for_attn=modified_v_for_attn,
                    selected_key_positions=selected_key_positions,
                    query_len=int(q_for_attn.shape[-2]),
                    gamma=float(intervention_value),
                    temperature=_parse_outro_temperature(sink_config["intervention"]),
                    head_mask=head_mask,
                )
            att = att.transpose(1, 2).contiguous().view(B, T, C)
            return block_module.attn_out(att), present

        return patched_attention

    for layer_idx in layers_to_patch:
        block = blocks[layer_idx]
        original_methods[layer_idx] = block.attention
        layer_name = f"layer_{layer_idx}"
        block.attention = MethodType(make_patched_attention(block, layer_name, original_methods[layer_idx]), block)

    try:
        yield {"layer_state": layer_state}
    finally:
        for layer_idx, original_attention in original_methods.items():
            blocks[layer_idx].attention = original_attention
