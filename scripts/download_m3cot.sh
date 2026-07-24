#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/nlp/zhz/PostVRG"
HF_CACHE_ROOT="${PROJECT_ROOT}/data/hf_cache"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

source /home/nlp/anaconda3/etc/profile.d/conda.sh
conda activate lavida

export HF_ENDPOINT
export HF_HOME="${HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"

mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

python - <<'PY'
import datasets
from pathlib import Path

dataset_name = "LightChen2333/M3CoT"
cache_root = Path("/media/nlp/zhz/PostVRG/data/hf_cache")
dataset_cache = cache_root / "datasets"
required_columns = {
    "id",
    "image",
    "question",
    "choices",
    "answer",
    "domain",
    "topic",
}

print(f"Downloading {dataset_name}")
print(f"HF cache root: {cache_root}")
print(f"HF endpoint: {__import__('os').environ.get('HF_ENDPOINT')}")
print(f"datasets version: {datasets.__version__}")

loaded = {}
for split in ("train", "validation", "test"):
    ds = datasets.load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(dataset_cache),
    )
    loaded[split] = ds
    missing = sorted(required_columns.difference(ds.column_names))
    if missing:
        raise RuntimeError(f"{split} split is missing required columns: {missing}")
    print(f"{split}: {len(ds)} rows; columns={ds.column_names}")

sample = loaded["test"][0]
if sample.get("image") is None:
    raise RuntimeError("test[0]['image'] is None; image data was not loaded correctly.")

print("M3CoT download and verification complete.")
print()
print("Use this when running PostVRG:")
print("  export HF_ENDPOINT=https://hf-mirror.com")
print("  export HF_HOME=/media/nlp/zhz/PostVRG/data/hf_cache")
print("  export HF_DATASETS_CACHE=/media/nlp/zhz/PostVRG/data/hf_cache/datasets")
print("  export HF_HUB_CACHE=/media/nlp/zhz/PostVRG/data/hf_cache/hub")
PY

cat <<'EOF'

Smoke-test command:
  source /home/nlp/anaconda3/etc/profile.d/conda.sh
  conda activate lavida
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/media/nlp/zhz/PostVRG/data/hf_cache
  export HF_DATASETS_CACHE=/media/nlp/zhz/PostVRG/data/hf_cache/datasets
  export HF_HUB_CACHE=/media/nlp/zhz/PostVRG/data/hf_cache/hub
  cd /media/nlp/zhz/PostVRG
  python M3CoT/PostVRG/postvrg_final.py \
    --limit 1 \
    --pretrained /media/nlp/zhz/Lavida/weight/lavida_reason \
    --vision-tower /media/nlp/zhz/Lavida/weight/siglip \
    --output-dir M3CoT/PostVRG/outputs/smoke_test
EOF
