#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/nlp/zhz/PostVRG}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${PROJECT_ROOT}/data/hf_cache}"
VSTAR_IMAGE_ROOT="${VSTAR_IMAGE_ROOT:-${PROJECT_ROOT}/data/vstar_bench}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${PROJECT_ROOT}"

source /home/nlp/anaconda3/etc/profile.d/conda.sh
conda activate lavida

export HF_ENDPOINT
export HF_HOME="${HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"

# This script must be allowed to contact the Hub/mirror.
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${VSTAR_IMAGE_ROOT}"

python - <<'PY'
import os
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/media/nlp/zhz/PostVRG"))
VSTAR_IMAGE_ROOT = Path(os.environ.get("VSTAR_IMAGE_ROOT", PROJECT_ROOT / "data" / "vstar_bench"))
FORCE = os.environ.get("FORCE", "").lower() in {"1", "true", "yes"}


def token_kwargs():
    token = os.environ.get("HF_TOKEN")
    use_login_token = os.environ.get("HF_USE_LOGIN_TOKEN", "").lower() in {"1", "true", "yes"}
    if token:
        return {"token": token}
    if use_login_token:
        return {"token": True}
    return {}


print("=" * 80)
print("Checking cached V*Bench rows")
print(f"  endpoint={os.environ.get('HF_ENDPOINT')}")
print(f"  image_root={VSTAR_IMAGE_ROOT}")
print(f"  force_download={FORCE}")

raw = datasets.load_dataset("craigwu/vstar_bench", split="test", **token_kwargs())

missing = []
for index, row in enumerate(raw):
    image = row.get("image") or row.get("img") or row.get("picture")
    if not image:
        raise RuntimeError(f"V*Bench row {index} has no image field: {row}")
    rel_path = str(image)
    if not (VSTAR_IMAGE_ROOT / rel_path).exists() or FORCE:
        missing.append((index, rel_path))

if not missing:
    print(f"All V*Bench images already exist: {len(raw)}/{len(raw)}")
else:
    print(f"Need to download {len(missing)} missing/stale image files")
    for index, rel_path in missing:
        print(f"  [{index}] {rel_path}")

    print("=" * 80)
    print("Downloading missing V*Bench images")
    for index, rel_path in missing:
        local_path = hf_hub_download(
            repo_id="craigwu/vstar_bench",
            repo_type="dataset",
            filename=rel_path,
            local_dir=str(VSTAR_IMAGE_ROOT),
            force_download=FORCE,
            **token_kwargs(),
        )
        print(f"  [{index}] {rel_path} -> {local_path}")

remaining = []
for index, row in enumerate(raw):
    rel_path = str(row.get("image") or row.get("img") or row.get("picture"))
    if not (VSTAR_IMAGE_ROOT / rel_path).exists():
        remaining.append((index, rel_path))

print("=" * 80)
if remaining:
    print(f"Still missing {len(remaining)} V*Bench images:")
    for index, rel_path in remaining:
        print(f"  [{index}] {rel_path}")
    raise SystemExit(1)

print(f"V*Bench image verification passed: {len(raw)}/{len(raw)}")
print("PostVRG path is ready:")
print(f"  --image-root {VSTAR_IMAGE_ROOT}")
PY
