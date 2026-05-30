import copy
import json
import logging
import os
import time
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.llava_llada import Llava_Llada
from lmms_eval.models.model_utils.load_video import read_video_pyav

from Scale_Attention.reweight_patch import (
    MASK_TOKEN_ID,
    build_fine_category_weights,
    build_prefix_from_multimodal_inputs,
    generate_with_dynamic_category_reweighting,
    get_special_token_ids,
    patch_category_reweight_attention,
)

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
except ImportError as e:
    raise ImportError(f"Failed to import LLaVA components for llava_llada_reweight: {e}")


eval_logger = logging.getLogger("lmms-eval")
DEBUG_PRINT_OUTPUT = os.environ.get("DEBUG_PRINT_OUTPUT", False)
DEBUG_LOAD_TRAINER = os.environ.get("DEBUG_LOAD_TRAINER", False)


def parse_bool_like(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@register_model("llava_llada_reweight")
class Llava_Llada_Reweight(Llava_Llada):
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "llava_llada",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config: Optional[str] = None,
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        video_decode_backend: str = "decord",
        mc_num=16,
        reweight_enable: Union[bool, str] = False,
        reweight_alpha_prompt: float = 1.0,
        reweight_alpha_visual: float = 1.0,
        reweight_alpha_generated: float = 1.0,
        reweight_alpha_mask: Optional[float] = None,
        reweight_alpha_normal: Optional[float] = None,
        reweight_alpha_special: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.reweight_enable = parse_bool_like(reweight_enable)
        self.reweight_alpha_prompt = float(reweight_alpha_prompt)
        self.reweight_alpha_visual = float(reweight_alpha_visual)
        self.reweight_alpha_generated = float(reweight_alpha_generated)
        self.reweight_alpha_mask = self.reweight_alpha_generated if reweight_alpha_mask is None else float(reweight_alpha_mask)
        self.reweight_alpha_normal = self.reweight_alpha_generated if reweight_alpha_normal is None else float(reweight_alpha_normal)
        self.reweight_alpha_special = self.reweight_alpha_generated if reweight_alpha_special is None else float(reweight_alpha_special)
        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            customized_config=customized_config,
            max_frames_num=max_frames_num,
            mm_spatial_pool_stride=mm_spatial_pool_stride,
            mm_spatial_pool_mode=mm_spatial_pool_mode,
            token_strategy=token_strategy,
            video_decode_backend=video_decode_backend,
            mc_num=mc_num,
            **kwargs,
        )

    def _generate_with_optional_reweight(self, input_ids, attention_masks, pad_token_ids, image_tensor, gen_kwargs):
        if not self.reweight_enable or image_tensor is None:
            return self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                use_cache=self.use_cache,
                **gen_kwargs,
            )

        prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
            model=self.model,
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=gen_kwargs.get("image_sizes", None),
            attention_mask=attention_masks,
        )
        special_token_ids = get_special_token_ids(self.tokenizer)
        initial_weights, _initial_meta = build_fine_category_weights(
            prefix_input_ids_full=prefix_input_ids_full,
            gen_tokens=torch.full(
                (int(gen_kwargs["max_new_tokens"]),),
                MASK_TOKEN_ID,
                dtype=torch.long,
                device=input_ids.device,
            ),
            special_token_ids=special_token_ids,
            alpha_prompt=self.reweight_alpha_prompt,
            alpha_visual=self.reweight_alpha_visual,
            alpha_mask=self.reweight_alpha_mask,
            alpha_normal=self.reweight_alpha_normal,
            alpha_special=self.reweight_alpha_special,
        )
        category_weight_state = {"weights": initial_weights}

        block_length = int(gen_kwargs.get("block_length", min(128, int(gen_kwargs["max_new_tokens"]))))
        step_ratio = gen_kwargs.get("step_ratio", None)
        if step_ratio is None and "step_per_block" in gen_kwargs:
            step_ratio = float(gen_kwargs["step_per_block"]) / float(block_length)
        if step_ratio is None:
            step_ratio = 1.0

        schedule_kwargs = gen_kwargs.get("schedule_kwargs", None) or {}
        schedule_shift = float(schedule_kwargs.get("shift", 1.0 / 3.0))
        with patch_category_reweight_attention(self.model, category_weight_state):
            sequences, _last_step_meta, _final_meta = generate_with_dynamic_category_reweighting(
                core_model=self.model.get_model(),
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                category_weight_state=category_weight_state,
                special_token_ids=special_token_ids,
                max_new_tokens=int(gen_kwargs["max_new_tokens"]),
                block_length=block_length,
                temperature=float(gen_kwargs.get("temperature", 0.0)),
                remasking=gen_kwargs.get("remasking", "low_confidence"),
                schedule=gen_kwargs.get("schedule", "none"),
                schedule_shift=schedule_shift,
                step_ratio=float(step_ratio),
                alpha_prompt=self.reweight_alpha_prompt,
                alpha_visual=self.reweight_alpha_visual,
                alpha_mask=self.reweight_alpha_mask,
                alpha_normal=self.reweight_alpha_normal,
                alpha_special=self.reweight_alpha_special,
            )
        return sequences

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        metadata = requests[0].metadata
        if DEBUG_PRINT_OUTPUT:
            re_ords = utils.Collator([reg.args for reg in requests], lambda x: x[-3], grouping=True)
        else:
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)
        if DEBUG_LOAD_TRAINER:
            ckpt1 = torch.load(DEBUG_LOAD_TRAINER, map_location="cpu")
            ckpt1 = {k.replace("module.model", "model"): v for k, v in ckpt1.items()}
            _res = self.model.load_state_dict(ckpt1, strict=False)
            print(f"DEBUG_LOAD_TRAINER:{DEBUG_LOAD_TRAINER} {_res}")
            print("Something is broken if above line does not show all keys matched!!!")
            del ckpt1

        delta_t = 0
        num_generated = 0
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            for visual, context in zip(batched_visuals, batched_contexts):
                t0 = time.time()
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")
                if self.overwrite_image_aspect:
                    self._config.image_aspect_ratio = self.overwrite_image_aspect

                if visual is None or visual == []:
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1
                    elif isinstance(visual[0], PIL.Image.Image):
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1
                    elif type(visual[0]) == str:
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num)
                            else:
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                if "llama_3" in self.conv_template or "llada" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)
                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            schedule_kwargs = {}
            for key in list(gen_kwargs.keys()):
                if key.startswith("schedule__"):
                    value = gen_kwargs.pop(key)
                    schedule_kwargs[key.replace("schedule__", "")] = value
            if len(schedule_kwargs) > 0:
                gen_kwargs["schedule_kwargs"] = schedule_kwargs

            if "block_length" not in gen_kwargs:
                gen_kwargs["block_length"] = min(128, gen_kwargs["max_new_tokens"])
            if "step_per_block" not in gen_kwargs and "step_ratio" not in gen_kwargs:
                gen_kwargs["step_per_block"] = gen_kwargs["block_length"]
            gen_kwargs["temperature"] = 0

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")

            try:
                with torch.inference_mode():
                    cont = self._generate_with_optional_reweight(
                        input_ids=input_ids,
                        attention_masks=attention_masks,
                        pad_token_ids=pad_token_ids,
                        image_tensor=image_tensor,
                        gen_kwargs=gen_kwargs,
                    )

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                text_outputs = [text_output.lstrip("!") for text_output in text_outputs]
            except Exception as e:
                raise e

            t1 = time.time()
            delta_t += t1 - t0
            num_generated += 1
            print(f"Avg Latency (of {num_generated}): {delta_t/num_generated}")
            if DEBUG_PRINT_OUTPUT:
                print(f"\n--------Start of Sample {batched_doc_id[0]}---------")
                print("Question: ", prompt_question)
                print("Answer: ", text_outputs)
                print("Answer: ", gen_kwargs)
                print("--------End---------")

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
