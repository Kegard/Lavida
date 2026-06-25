import logging
from contextlib import contextmanager
from typing import Any, List, Union

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.llava_llada import Llava_Llada

from Sink.sink_patch import (
    build_sink_config,
    infer_multimodal_layout,
    patch_attention_sink,
    resolve_layer_indices,
)


eval_logger = logging.getLogger("lmms-eval")


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


@register_model("llava_llada_sink")
class Llava_Llada_Sink(Llava_Llada):
    def __init__(
        self,
        *args,
        sink_enable: Union[bool, str] = False,
        sink_intervention: str = "none",
        sink_selector: str = "top_attn",
        sink_topk: int = 1,
        sink_layers: str = "last",
        sink_steps: str = "both",
        sink_query_scope: str = "text",
        sink_control: str = "none",
        sink_head_scope: str = "all",
        sink_seed: int = 0,
        sink_debug: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sink_enabled = _as_bool(sink_enable)
        self._sink_layers_spec = str(sink_layers)
        self._sink_config = build_sink_config(
            enabled=sink_enable,
            intervention=sink_intervention,
            selector=sink_selector,
            topk=sink_topk,
            steps=sink_steps,
            query_scope=sink_query_scope,
            control=sink_control,
            head_scope=sink_head_scope,
            seed=sink_seed,
            debug=sink_debug,
        )
        if self._sink_enabled:
            eval_logger.warning(
                "Enable sink intervention: intervention=%s selector=%s topk=%s layers=%s steps=%s query_scope=%s control=%s head_scope=%s",
                self._sink_config["intervention"],
                self._sink_config["selector"],
                self._sink_config["topk"],
                self._sink_layers_spec,
                ",".join(self._sink_config["step_modes"]),
                self._sink_config["query_scope"],
                self._sink_config["control"],
                self._sink_config["head_scope"],
            )

    @contextmanager
    def _sink_runtime_context(self):
        if not self._sink_enabled or self._sink_config["intervention"] == "none":
            yield None
            return

        model_obj = self.model
        original_generate = model_obj.generate
        layer_indices = resolve_layer_indices(
            len(model_obj.get_model().transformer.blocks),
            self._sink_layers_spec,
        )

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
                layout = infer_multimodal_layout(model_obj, input_ids, images, image_sizes)
            except Exception as e:
                eval_logger.warning(f"Sink layout inference failed, fallback to normal generate: {e}")
                return original_generate(*args, **kwargs)

            with patch_attention_sink(
                model=model_obj,
                layers_to_patch=layer_indices,
                visual_positions=layout["visual_positions"],
                text_query_positions=layout["text_query_positions"],
                sink_config=self._sink_config,
            ):
                return original_generate(*args, **kwargs)

        model_obj.generate = wrapped_generate
        try:
            yield None
        finally:
            model_obj.generate = original_generate

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if not self._sink_enabled or self._sink_config["intervention"] == "none":
            return super().generate_until(requests)

        with self._sink_runtime_context():
            return super().generate_until(requests)
