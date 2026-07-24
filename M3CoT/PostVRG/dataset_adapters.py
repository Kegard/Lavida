"""Dataset adapters for running PostVRG on non-M3CoT multiple-choice benchmarks.

The PostVRG runner expects an M3CoT-like row:
  id, image, context, question, choices, answer, domain, topic.

This module keeps the decoding/evaluation code unchanged by normalizing other
benchmarks into that row format at load time.
"""

import argparse
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import datasets
from PIL import Image


LETTERS = "ABCDEFG"


def add_dataset_adapter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        default="m3cot",
        choices=[
            "m3cot",
            "scienceqa_img",
            "mmbench_en_dev",
            "mmbench_en_dev_lite",
            "vstar",
        ],
        help="Benchmark adapter to use. All adapters emit M3CoT-like rows.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional HuggingFace dataset config/name override.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to datasets.load_dataset.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HuggingFace token value; use 'true' to read the logged-in token.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root directory for datasets whose image column stores relative paths.",
    )


class PostVRGDataset:
    """Small lazy adapter around a HuggingFace Dataset.

    It implements the subset of the datasets.Dataset API used by postvrg_final:
    iteration, len(), filter(), shuffle(), and select().
    """

    def __init__(
        self,
        raw_dataset,
        adapter: Callable[[Dict[str, Any], int], Dict[str, Any]],
        indices: Optional[Sequence[int]] = None,
    ):
        self.raw_dataset = raw_dataset
        self.adapter = adapter
        self.indices = list(range(len(raw_dataset))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        for raw_index in self.indices:
            yield self.adapter(self.raw_dataset[int(raw_index)], int(raw_index))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_index = self.indices[int(index)]
        return self.adapter(self.raw_dataset[int(raw_index)], int(raw_index))

    def filter(self, fn: Callable[[Dict[str, Any]], bool]):
        kept = []
        for raw_index in self.indices:
            row = self.adapter(self.raw_dataset[int(raw_index)], int(raw_index))
            if fn(row):
                kept.append(raw_index)
        return PostVRGDataset(self.raw_dataset, self.adapter, kept)

    def shuffle(self, seed: int):
        shuffled = list(self.indices)
        random.Random(seed).shuffle(shuffled)
        return PostVRGDataset(self.raw_dataset, self.adapter, shuffled)

    def select(self, indices: Iterable[int]):
        selected = []
        for index in indices:
            index = int(index)
            if index < 0 or index >= len(self.indices):
                raise IndexError(f"Index {index} out of range for dataset of size {len(self.indices)}")
            selected.append(self.indices[index])
        return PostVRGDataset(self.raw_dataset, self.adapter, selected)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _clean_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _first_present(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and not _is_missing(row[key]):
            return row[key]
    return default


def _choice_list_from_dict(value: Dict[str, Any]) -> List[str]:
    choices = []
    for key in LETTERS:
        if key in value and not _is_missing(value[key]):
            choices.append(_clean_text(value[key]))
    if choices:
        return choices
    return [_clean_text(value[key]) for key in sorted(value) if not _is_missing(value[key])]


def extract_choices(row: Dict[str, Any]) -> List[str]:
    for key in ("choices", "options", "candidates"):
        value = row.get(key)
        if isinstance(value, dict):
            choices = _choice_list_from_dict(value)
            if choices:
                return choices
        if isinstance(value, (list, tuple)):
            choices = [_clean_text(item) for item in value if not _is_missing(item)]
            if choices:
                return choices

    choices = []
    for key in LETTERS:
        if key in row and not _is_missing(row[key]):
            choices.append(_clean_text(row[key]))
    return choices


def parse_embedded_mcq_text(text: Any) -> tuple[str, List[str]]:
    text = _clean_text(text)
    if not text:
        return "", []
    text = re.sub(
        r"\s*Answer with the option'?s letter from the given choices directly\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    matches = list(re.finditer(r"\(([A-Ga-g])\)", text))
    if not matches:
        return text, []

    question = text[: matches[0].start()].strip()
    choices = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        choice = text[start:end].strip()
        if choice:
            choices.append(choice)
    return question, choices


def normalize_answer(answer: Any, choices: Optional[Sequence[str]] = None) -> str:
    if isinstance(answer, bool):
        return str(answer)
    if isinstance(answer, int):
        if 0 <= answer < len(LETTERS):
            return LETTERS[answer]
        return str(answer)

    text = _clean_text(answer)
    if not text:
        return "FAILED"

    if choices:
        lowered = text.lower()
        for index, choice in enumerate(choices):
            if lowered == _clean_text(choice).lower() and index < len(LETTERS):
                return LETTERS[index]

    if text.isdigit():
        value = int(text)
        if 0 <= value < len(LETTERS):
            return LETTERS[value]

    valid = LETTERS[: len(choices)] if choices else LETTERS
    patterns = [
        rf"^\s*([{valid}{valid.lower()}])\s*$",
        rf"^\s*\(?\s*([{valid}{valid.lower()}])\s*\)?[\.:]?\s*$",
        rf"\banswer\s*[:：]\s*\(?\s*([{valid}{valid.lower()}])\s*\)?",
        rf"\(([{valid}{valid.lower()}])\)",
        rf"\b([{valid}{valid.lower()}])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    return text.upper()


def adapt_m3cot(row: Dict[str, Any], raw_index: int) -> Dict[str, Any]:
    choices = extract_choices(row)
    return {
        "id": _clean_text(row.get("id")) or f"m3cot-{raw_index}",
        "image": row.get("image"),
        "context": _clean_text(row.get("context")),
        "question": _clean_text(row.get("question")),
        "choices": choices,
        "answer": normalize_answer(row.get("answer"), choices),
        "domain": _clean_text(row.get("domain")) or "m3cot",
        "topic": _clean_text(row.get("topic")) or "m3cot",
        "benchmark": "m3cot",
        "raw_index": raw_index,
    }


def adapt_scienceqa_img(row: Dict[str, Any], raw_index: int) -> Dict[str, Any]:
    choices = extract_choices(row)
    topic = _first_present(row, ("topic", "subject", "category", "skill"), "scienceqa_img")
    return {
        "id": _clean_text(_first_present(row, ("id", "pid", "question_id"), f"scienceqa-img-{raw_index}")),
        "image": row.get("image"),
        "context": _clean_text(_first_present(row, ("context", "hint"), "")),
        "question": _clean_text(row.get("question")),
        "choices": choices,
        "answer": normalize_answer(row.get("answer"), choices),
        "domain": "scienceqa_img",
        "topic": _clean_text(topic) or "scienceqa_img",
        "benchmark": "scienceqa_img",
        "raw_index": raw_index,
    }


def adapt_mmbench(row: Dict[str, Any], raw_index: int) -> Dict[str, Any]:
    choices = extract_choices(row)
    category = _first_present(row, ("category", "l2-category", "L2-category"), "mmbench")
    topic = _first_present(row, ("L2-category", "l2-category", "category"), category)
    return {
        "id": _clean_text(_first_present(row, ("id", "index"), f"mmbench-{raw_index}")),
        "image": row.get("image"),
        "context": _clean_text(row.get("hint")),
        "question": _clean_text(row.get("question")),
        "choices": choices,
        "answer": normalize_answer(row.get("answer"), choices),
        "domain": "mmbench",
        "topic": _clean_text(topic) or "mmbench",
        "benchmark": "mmbench",
        "raw_index": raw_index,
        "category": _clean_text(category),
        "mmbench_index": row.get("index"),
    }


def adapt_vstar(row: Dict[str, Any], raw_index: int) -> Dict[str, Any]:
    choices = extract_choices(row)
    embedded_question, embedded_choices = parse_embedded_mcq_text(row.get("text"))
    if not choices and embedded_choices:
        choices = embedded_choices
    answer = _first_present(
        row,
        ("answer", "label", "gt_answer", "gt", "target", "correct_answer", "Answer"),
        None,
    )
    topic = _first_present(row, ("category", "task", "type", "question_type", "subtask"), "vstar")
    question = _clean_text(_first_present(row, ("question", "query", "prompt"), ""))
    if not question:
        question = embedded_question
    return {
        "id": _clean_text(_first_present(row, ("id", "index", "question_id"), f"vstar-{raw_index}")),
        "image": _first_present(row, ("image", "img", "picture")),
        "context": _clean_text(_first_present(row, ("context", "hint"), "")),
        "question": question,
        "choices": choices,
        "answer": normalize_answer(answer, choices),
        "domain": "vstar",
        "topic": _clean_text(topic) or "vstar",
        "benchmark": "vstar",
        "raw_index": raw_index,
    }


def resolve_image_value(image: Any, image_root: Optional[str] = None):
    if image is None or hasattr(image, "convert"):
        return image
    if isinstance(image, dict):
        for key in ("path", "image", "bytes"):
            if key in image:
                return resolve_image_value(image[key], image_root)
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        candidates = []
        if image_path.is_absolute():
            candidates.append(image_path)
        if image_root:
            candidates.append(Path(image_root) / image_path)
        candidates.append(image_path)
        for candidate in candidates:
            if candidate.exists():
                return Image.open(candidate).convert("RGB")
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Image path {image!r} is not a PIL image and was not found. "
            f"Searched: {searched}. Pass --image-root for this benchmark."
        )
    return image


def with_image_resolver(adapter, image_root: Optional[str] = None):
    def wrapped(row: Dict[str, Any], raw_index: int) -> Dict[str, Any]:
        adapted = adapter(row, raw_index)
        adapted["image"] = resolve_image_value(adapted.get("image"), image_root=image_root)
        return adapted

    return wrapped


ADAPTERS = {
    "m3cot": adapt_m3cot,
    "scienceqa_img": adapt_scienceqa_img,
    "mmbench_en_dev": adapt_mmbench,
    "mmbench_en_dev_lite": adapt_mmbench,
    "vstar": adapt_vstar,
}


def infer_benchmark(dataset_path: str) -> str:
    path = (dataset_path or "").lower()
    if "scienceqa" in path:
        return "scienceqa_img"
    if "mmbench" in path:
        return "mmbench_en_dev"
    if "vstar" in path or "v-star" in path:
        return "vstar"
    return "m3cot"


def _hf_token_arg(args) -> Optional[Any]:
    token = getattr(args, "hf_token", None)
    if token is None:
        return None
    if str(token).lower() in {"true", "1", "yes"}:
        return True
    if str(token).lower() in {"false", "0", "no"}:
        return False
    return token


def _load_dataset(path: str, name: Optional[str], split: str, args):
    kwargs = {}
    token = _hf_token_arg(args)
    if token is not None:
        kwargs["token"] = token
    if getattr(args, "trust_remote_code", False):
        kwargs["trust_remote_code"] = True
    if name:
        return datasets.load_dataset(path, name, split=split, **kwargs)
    return datasets.load_dataset(path, split=split, **kwargs)


def load_postvrg_dataset(args) -> PostVRGDataset:
    benchmark = getattr(args, "benchmark", None) or infer_benchmark(getattr(args, "dataset_path", ""))
    split = getattr(args, "split", None) or "test"
    dataset_name = getattr(args, "dataset_name", None)

    if benchmark == "m3cot":
        path = getattr(args, "dataset_path", None) or "LightChen2333/M3CoT"
        raw = _load_dataset(path, dataset_name, split, args)
    elif benchmark == "scienceqa_img":
        path = getattr(args, "dataset_path", None) or "lmms-lab/ScienceQA"
        name = dataset_name or "ScienceQA-IMG"
        raw = _load_dataset(path, name, split, args)
    elif benchmark == "mmbench_en_dev":
        path = getattr(args, "dataset_path", None) or "lmms-lab/MMBench"
        name = dataset_name or "en"
        raw = _load_dataset(path, name, split, args)
    elif benchmark == "mmbench_en_dev_lite":
        path = getattr(args, "dataset_path", None) or "lmms-lab/LMMs-Eval-Lite"
        name = dataset_name or "mmbench_en_dev"
        raw = _load_dataset(path, name, split, args)
    elif benchmark == "vstar":
        path = getattr(args, "dataset_path", None) or "craigwu/vstar_bench"
        raw = _load_dataset(path, dataset_name, split, args)
    else:
        raise ValueError(f"Unsupported benchmark adapter: {benchmark}")

    return PostVRGDataset(raw, with_image_resolver(ADAPTERS[benchmark], getattr(args, "image_root", None)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PostVRG dataset adapter.")
    add_dataset_adapter_args(parser)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=1)
    args = parser.parse_args()

    dataset = load_postvrg_dataset(args)
    print(f"adapted_size={len(dataset)}")
    for idx, row in enumerate(dataset):
        if idx >= args.num:
            break
        preview = dict(row)
        preview["image"] = None if row.get("image") is None else str(type(row["image"]))
        print(preview)


if __name__ == "__main__":
    main()
