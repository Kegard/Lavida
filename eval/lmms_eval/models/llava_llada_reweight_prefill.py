import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, Union

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.llava_llada import Llava_Llada

from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs
from Scale_Attention.reweight_prefill import (
    _infer_prefill_layout as _infer_prefill_layout_from_prefix,
    patch_attention_prefill_boost,
)

eval_logger = logging.getLogger("lmms-eval")


def _infer_prefill_layout(model_obj, input_ids, images, image_sizes):
    attention_mask = kwargs_attention_mask = None
    if input_ids is not None:
        attention_mask = input_ids.new_ones(input_ids.shape, dtype=input_ids.dtype)
        kwargs_attention_mask = attention_mask.to(dtype=getattr(input_ids, "dtype", None))

    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model_obj,
        input_ids=input_ids,
        images=images,
        image_sizes=image_sizes,
        attention_mask=kwargs_attention_mask,
    )
    del prefix_embeds
    return _infer_prefill_layout_from_prefix(prefix_input_ids_full)


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def _as_float(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return float(default)
        return float(s)
    return float(x)


@register_model("llava_llada_reweight_prefill")
class Llava_Llada_ReweightPrefill(Llava_Llada):
    def __init__(
        self,
        *args,
        reweight_prefill_enable: Union[bool, str] = False,
        reweight_prefill_gamma: float = 1.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reweight_prefill_enabled = _as_bool(reweight_prefill_enable)
        self._reweight_prefill_gamma = _as_float(reweight_prefill_gamma, 1.2)
        self._reweight_prefill_layers_to_patch = list(range(len(self.model.get_model().transformer.blocks)))
        if self._reweight_prefill_enabled:
            eval_logger.warning(f"Enable prefill reweight: gamma={self._reweight_prefill_gamma}")

    @contextmanager
    def _reweight_prefill_runtime_context(self):
        if not self._reweight_prefill_enabled:
            yield None
            return

        model_obj = self.model
        original_generate = model_obj.generate

        def wrapped_generate(*args, **kwargs):
            input_ids = args[0] if len(args) > 0 else kwargs.get("input_ids", kwargs.get("inputs", None))
            images = kwargs.get("images", None)
            image_sizes = kwargs.get("image_sizes", None)
            modalities = kwargs.get("modalities", ["image"])

            if (
                images is None
                or input_ids is None
                or (isinstance(modalities, list) and len(modalities) > 0 and modalities[0] != "image")
            ):
                return original_generate(*args, **kwargs)

            try:
                layout = _infer_prefill_layout(model_obj, input_ids, images, image_sizes)
                if layout is None:
                    return original_generate(*args, **kwargs)
                vis_start, vis_end, text_query_positions = layout
            except Exception as e:
                eval_logger.warning(f"Prefill reweight span inference failed, fallback to normal generate: {e}")
                return original_generate(*args, **kwargs)

            with patch_attention_prefill_boost(
                model=model_obj,
                layers_to_patch=self._reweight_prefill_layers_to_patch,
                vis_start=vis_start,
                vis_end=vis_end,
                text_query_positions=text_query_positions,
                gamma=float(self._reweight_prefill_gamma),
            ):
                return original_generate(*args, **kwargs)

        model_obj.generate = wrapped_generate
        try:
            yield None
        finally:
            model_obj.generate = original_generate

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if not self._reweight_prefill_enabled:
            return super().generate_until(requests)

        with self._reweight_prefill_runtime_context():
            return super().generate_until(requests)
