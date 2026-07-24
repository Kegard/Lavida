"""Dataset adapter bridge for MMaDA PostVRG evaluation.

The canonical adapters live in ``M3CoT/PostVRG/dataset_adapters.py`` so LaViDa
and MMaDA evaluations normalize benchmarks identically. This module gives
``MMaDA/mmada_postvrg.py`` a local import path while keeping one source of truth.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from M3CoT.PostVRG.dataset_adapters import add_dataset_adapter_args, load_postvrg_dataset


__all__ = ["add_dataset_adapter_args", "load_postvrg_dataset"]
