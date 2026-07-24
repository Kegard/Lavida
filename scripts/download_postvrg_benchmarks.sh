#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/nlp/zhz/PostVRG"
HF_CACHE_ROOT="${PROJECT_ROOT}/data/hf_cache"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${PROJECT_ROOT}"

source /home/nlp/anaconda3/etc/profile.d/conda.sh
conda activate lavida

export HF_ENDPOINT
export HF_HOME="${HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"

mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

python - <<'PY'
import os
import sys
from pathlib import Path

import datasets
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path("/media/nlp/zhz/PostVRG")
VSTAR_IMAGE_ROOT = PROJECT_ROOT / "data" / "vstar_bench"
sys.path.insert(0, str(PROJECT_ROOT))

from M3CoT.PostVRG.dataset_adapters import PostVRGDataset
from M3CoT.PostVRG.dataset_adapters import (
    adapt_mmbench,
    adapt_scienceqa_img,
    adapt_vstar,
    with_image_resolver,
)


def token_kwargs():
    token = os.environ.get("HF_TOKEN")
    use_login_token = os.environ.get("HF_USE_LOGIN_TOKEN", "").lower() in {"1", "true", "yes"}
    if token:
        return {"token": token}
    if use_login_token:
        return {"token": True}
    return {}


def load_and_verify(title, path, name, split, adapter):
    kwargs = token_kwargs()
    print("=" * 80)
    print(f"Downloading: {title}")
    print(f"  path={path}")
    print(f"  name={name}")
    print(f"  split={split}")
    print(f"  endpoint={os.environ.get('HF_ENDPOINT')}")
    print(f"  cache={os.environ.get('HF_DATASETS_CACHE')}")
    if name:
        raw = datasets.load_dataset(path, name, split=split, **kwargs)
    else:
        raw = datasets.load_dataset(path, split=split, **kwargs)

    image_root = str(VSTAR_IMAGE_ROOT) if title == "V*Bench" else None
    adapted = PostVRGDataset(raw, with_image_resolver(adapter, image_root=image_root))
    first = adapted[0]
    required = ["id", "image", "context", "question", "choices", "answer", "domain", "topic"]
    missing = [key for key in required if key not in first]
    if missing:
        raise RuntimeError(f"{title}: adapter missing required keys: {missing}")
    if first["image"] is None:
        raise RuntimeError(f"{title}: first adapted row has image=None")
    if not first["question"]:
        raise RuntimeError(f"{title}: first adapted row has empty question")
    if not first["choices"]:
        raise RuntimeError(f"{title}: first adapted row has empty choices")
    if not first["answer"]:
        raise RuntimeError(f"{title}: first adapted row has empty answer")

    print(f"  rows={len(raw)}")
    print(f"  raw_columns={raw.column_names}")
    print(
        "  adapted_first="
        f"id={first['id']!r}, domain={first['domain']!r}, topic={first['topic']!r}, "
        f"choices={first['choices']!r}, answer={first['answer']!r}"
    )
    return raw


jobs = [
    {
        "title": "ScienceQA IMG",
        "path": "lmms-lab/ScienceQA",
        "name": "ScienceQA-IMG",
        "split": "test",
        "adapter": adapt_scienceqa_img,
    },
    {
        "title": "MMBench EN dev",
        "path": "lmms-lab/MMBench",
        "name": "en",
        "split": "dev",
        "adapter": adapt_mmbench,
    },
    {
        "title": "MMBench EN dev lite",
        "path": "lmms-lab/LMMs-Eval-Lite",
        "name": "mmbench_en_dev",
        "split": "lite",
        "adapter": adapt_mmbench,
    },
    {
        "title": "V*Bench",
        "path": "craigwu/vstar_bench",
        "name": None,
        "split": "test",
        "adapter": adapt_vstar,
    },
]

print(f"datasets version: {datasets.__version__}")
print("=" * 80)
print("Downloading V*Bench image files")
snapshot_download(
    repo_id="craigwu/vstar_bench",
    repo_type="dataset",
    local_dir=str(VSTAR_IMAGE_ROOT),
    allow_patterns=[
        "direct_attributes/*.jpg",
        "direct_attributes/*.json",
        "relative_position/*.jpg",
        "relative_position/*.json",
        "OCR/*.jpg",
        "OCR/*.json",
        "GPT4V-hard/*.jpg",
        "GPT4V-hard/*.json",
    ],
    **token_kwargs(),
)
print(f"V*Bench image root: {VSTAR_IMAGE_ROOT}")

for job in jobs:
    load_and_verify(**job)

print("=" * 80)
print("All requested PostVRG benchmark datasets downloaded and adapter-verified.")
print()
print("Run PostVRG with the same cache variables:")
print("  export HF_HOME=/media/nlp/zhz/PostVRG/data/hf_cache")
print("  export HF_DATASETS_CACHE=/media/nlp/zhz/PostVRG/data/hf_cache/datasets")
print("  export HF_HUB_CACHE=/media/nlp/zhz/PostVRG/data/hf_cache/hub")
print("  export HF_HUB_OFFLINE=1")
print("  export HF_DATASETS_OFFLINE=1")
print("  export VSTAR_IMAGE_ROOT=/media/nlp/zhz/PostVRG/data/vstar_bench")
PY

cat <<'EOF'

Examples after download:

  python M3CoT/PostVRG/dataset_adapters.py --benchmark scienceqa_img --split test --num 1
  python M3CoT/PostVRG/dataset_adapters.py --benchmark mmbench_en_dev --split dev --num 1
  python M3CoT/PostVRG/dataset_adapters.py --benchmark mmbench_en_dev_lite --split lite --num 1
  python M3CoT/PostVRG/dataset_adapters.py --benchmark vstar --split test --image-root data/vstar_bench --num 1

If a dataset requires HuggingFace auth, either run:
  huggingface-cli login
  HF_USE_LOGIN_TOKEN=1 ./scripts/download_postvrg_benchmarks.sh

or pass a token directly:
  HF_TOKEN=hf_xxx ./scripts/download_postvrg_benchmarks.sh
EOF
