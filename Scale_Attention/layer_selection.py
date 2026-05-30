import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def infer_square_grid(num_visual_tokens: int) -> Tuple[int, int]:
    side = int(math.sqrt(int(num_visual_tokens)))
    if side * side != int(num_visual_tokens):
        raise ValueError(f"Expected a square visual-token grid, got {num_visual_tokens} visual tokens.")
    return side, side


def compute_laplacian_energy(visual_map: torch.Tensor) -> float:
    if visual_map.ndim != 2:
        raise ValueError(f"visual_map must be 2D, got shape={tuple(visual_map.shape)}")

    values = visual_map.to(dtype=torch.float32)
    total = values.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    values = values / total.clamp_min(1e-12)

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=values.dtype,
        device=values.device,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(values.view(1, 1, *values.shape), kernel, padding=1)
    return float((laplacian.square().mean()).item())


def compute_visual_layer_metrics(
    attn_probs: torch.Tensor,
    visual_mask: torch.Tensor,
    selected_queries: torch.Tensor,
    grid_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    selected_queries = selected_queries.detach().cpu().to(dtype=torch.long).view(-1)
    visual_mask = visual_mask.detach().cpu().to(dtype=torch.bool).view(-1)
    if selected_queries.numel() == 0 or visual_mask.sum().item() == 0:
        h, w = grid_size if grid_size is not None else (0, 0)
        return {
            "visual_mass": 0.0,
            "laplacian_energy": 0.0,
            "selection_score": 0.0,
            "visual_map": torch.zeros((h, w), dtype=torch.float32),
        }

    probs = attn_probs.detach().cpu().to(dtype=torch.float32)
    if probs.ndim != 4:
        raise ValueError(f"attn_probs must have shape [B, H, Q, K], got {tuple(probs.shape)}")

    max_query = int(probs.shape[2])
    selected_queries = selected_queries[(selected_queries >= 0) & (selected_queries < max_query)]
    if selected_queries.numel() == 0:
        h, w = grid_size if grid_size is not None else infer_square_grid(int(visual_mask.sum().item()))
        return {
            "visual_mass": 0.0,
            "laplacian_energy": 0.0,
            "selection_score": 0.0,
            "visual_map": torch.zeros((h, w), dtype=torch.float32),
        }

    prefix_len = int(visual_mask.shape[0])
    visual_probs = probs[0, :, selected_queries, :prefix_len][:, :, visual_mask]
    visual_map_flat = visual_probs.mean(dim=(0, 1))
    visual_mass = float(visual_map_flat.sum().item())

    if grid_size is None:
        grid_size = infer_square_grid(int(visual_mask.sum().item()))
    h, w = grid_size
    visual_map = visual_map_flat.view(h, w)
    laplacian_energy = compute_laplacian_energy(visual_map)
    return {
        "visual_mass": visual_mass,
        "laplacian_energy": laplacian_energy,
        "selection_score": visual_mass * laplacian_energy,
        "visual_map": visual_map,
    }


def collect_layer_selection_metrics(
    attention_store: Dict[str, List[torch.Tensor]],
    visual_mask: torch.Tensor,
    selected_queries: torch.Tensor,
    layers: Iterable[int],
    grid_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, object]]:
    if grid_size is None:
        grid_size = infer_square_grid(int(visual_mask.detach().cpu().to(dtype=torch.bool).sum().item()))

    metrics = []
    for layer_idx in layers:
        layer_key = f"layer_{layer_idx}"
        if layer_key not in attention_store or not attention_store[layer_key]:
            layer_metrics = {
                "layer": int(layer_idx),
                "visual_mass": 0.0,
                "laplacian_energy": 0.0,
                "selection_score": 0.0,
                "visual_map": torch.zeros(grid_size, dtype=torch.float32),
            }
        else:
            layer_metrics = compute_visual_layer_metrics(
                attention_store[layer_key][0],
                visual_mask=visual_mask,
                selected_queries=selected_queries,
                grid_size=grid_size,
            )
            layer_metrics["layer"] = int(layer_idx)
        metrics.append(layer_metrics)
    return metrics


def select_recovery_and_emergence_layers(
    layer_metrics: List[Dict[str, object]],
    min_visual_mass: float = 1e-4,
) -> Dict[str, object]:
    candidates = [
        item for item in layer_metrics
        if float(item["visual_mass"]) >= float(min_visual_mass)
    ]
    if not candidates:
        candidates = layer_metrics
    if not candidates:
        raise ValueError("No layer metrics were provided.")

    recovery = max(candidates, key=lambda item: float(item["selection_score"]))
    recovery_layer = int(recovery["layer"])
    later = [item for item in candidates if int(item["layer"]) > recovery_layer]

    if later:
        ordered = sorted(candidates, key=lambda item: int(item["layer"]))
        score_by_layer = {int(item["layer"]): float(item["selection_score"]) for item in ordered}
        drop_candidates = []
        for item in later:
            layer = int(item["layer"])
            prev_layers = [prev for prev in score_by_layer if prev < layer]
            if not prev_layers:
                continue
            prev_layer = max(prev_layers)
            drop = score_by_layer[prev_layer] - float(item["selection_score"])
            drop_candidates.append((drop, item))
        if drop_candidates:
            emergence = max(drop_candidates, key=lambda pair: pair[0])[1]
        else:
            emergence = min(later, key=lambda item: float(item["selection_score"]))
    else:
        before = [item for item in candidates if int(item["layer"]) < recovery_layer]
        emergence = min(before, key=lambda item: float(item["selection_score"])) if before else recovery

    return {
        "recovery_layer": int(recovery["layer"]),
        "emergence_layer": int(emergence["layer"]),
        "recovery_score": float(recovery["selection_score"]),
        "emergence_score": float(emergence["selection_score"]),
        "recovery_visual_mass": float(recovery["visual_mass"]),
        "emergence_visual_mass": float(emergence["visual_mass"]),
        "recovery_laplacian_energy": float(recovery["laplacian_energy"]),
        "emergence_laplacian_energy": float(emergence["laplacian_energy"]),
    }


def strip_visual_maps(layer_metrics: List[Dict[str, object]]) -> List[Dict[str, float]]:
    stripped = []
    for item in layer_metrics:
        stripped.append(
            {
                "layer": int(item["layer"]),
                "visual_mass": float(item["visual_mass"]),
                "laplacian_energy": float(item["laplacian_energy"]),
                "selection_score": float(item["selection_score"]),
            }
        )
    return stripped
