#!/usr/bin/env python3
"""
Single-run entry point for the prefill-sink lmms-eval adapter.

Use `--prefill-sink-enabled False` for baseline runs and `True` for sink runs.
The actual sink selection / penalty logic still lives in the lmms-eval model
adapter (llava_llada_prefill_sink).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FALLBACK_EVAL_SCRIPT = Path("/data/jindong_gu/Failed_LaViDa/eval/run_lmms_eval_prefill_sink.py")


def env_default(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return fallback if value is None or value == "" else value


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one lmms-eval pass with optional prefill-sink behavior."
    )
    parser.add_argument("model_path", help="Path to the pretrained LaViDa checkpoint.")
    parser.add_argument(
        "--eval-script",
        default="",
        help="Path to run_lmms_eval_prefill_sink.py. Defaults to the local eval/ copy, then the Failed_LaViDa copy.",
    )
    parser.add_argument("--tasks", default=env_default("TASKS", "mmmu_val"))
    parser.add_argument(
        "--gen-args",
        default=env_default(
            "GEN_ARGS",
            "prefix_lm=True,max_new_tokens=32,step_ratio=0.5,schedule=shift,schedule__shift=0.33",
        ),
    )
    parser.add_argument("--out-base", default=env_default("OUT_BASE", "logs/prefill_sink_eval"))
    parser.add_argument(
        "--prefill-sink-enabled",
        default=env_default("PREFILL_SINK_ENABLED", "False"),
        help="Enable sink selection and decode penalty for this single run.",
    )
    parser.add_argument(
        "--prefill-sink-common-args",
        default=env_default(
            "PREFILL_SINK_COMMON_ARGS",
            "prefill_sink_topk=4,prefill_sink_prefill_calls=1,prefill_sink_query_scope=text,"
            "prefill_sink_score_eps=1e-6,prefill_sink_var_weight=1.0,"
            "prefill_sink_log_events=False,prefill_sink_debug=False",
        ),
    )
    parser.add_argument(
        "--llada-vision-encoder",
        default=env_default("LLADA_VISION_ENCODER", "/data/jindong_gu/weight/siglip"),
    )
    parser.add_argument(
        "--llava-overwrite-image-aspect",
        default=env_default("LLAVA_OVERWRITE_IMAGE_ASPECT", "pad"),
    )
    parser.add_argument("--num-processes", type=int, default=int(env_default("NUM_PROCESSES", "1")))
    parser.add_argument("--num-machines", type=int, default=int(env_default("NUM_MACHINES", "1")))
    parser.add_argument("--machine-rank", type=int, default=int(env_default("MACHINE_RANK", "0")))
    parser.add_argument("--main-process-port", type=int, default=int(env_default("MAIN_PROCESS_PORT", "29551")))
    parser.add_argument(
        "--accelerate-binary",
        default=env_default("ACCELERATE_BINARY", "accelerate"),
        help="Accelerate launcher executable or module entry point.",
    )
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args
    return args


def resolve_eval_script(explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    local_candidate = REPO_ROOT / "eval" / "run_lmms_eval_prefill_sink.py"
    if local_candidate.exists():
        return local_candidate
    if DEFAULT_FALLBACK_EVAL_SCRIPT.exists():
        return DEFAULT_FALLBACK_EVAL_SCRIPT

    raise FileNotFoundError(
        "Could not find run_lmms_eval_prefill_sink.py in the local eval/ tree or in Failed_LaViDa."
    )


def build_command(
    *,
    accelerate_binary: str,
    eval_script: Path,
    model_path: str,
    tasks: str,
    gen_args: str,
    out_base: str,
    sink_enabled: bool,
    common_args: str,
    num_processes: int,
    num_machines: int,
    machine_rank: int,
    main_process_port: int,
    extra_args: Sequence[str],
) -> tuple[list[str], str, str]:
    model_args = build_model_args(model_path, sink_enabled, common_args)
    suffix = "prefill_sink_enabled" if sink_enabled else "prefill_sink_baseline"
    run_dir = "prefill_sink" if sink_enabled else "baseline"
    return [
        accelerate_binary,
        "launch",
        f"--num_processes={num_processes}",
        f"--num_machines={num_machines}",
        f"--machine_rank={machine_rank}",
        "--main_process_port",
        str(main_process_port),
        str(eval_script),
        "--model",
        "llava_llada_prefill_sink",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        "1",
        "--gen_kwargs",
        gen_args,
        "--log_samples",
        "--log_samples_suffix",
        suffix,
        "--output_path",
        str(Path(out_base) / run_dir),
        "--verbosity",
        "DEBUG",
        *extra_args,
    ], model_args, suffix


def build_model_args(model_path: str, sink_enabled: bool, common_args: str) -> str:
    sink_args = f"prefill_sink_enabled={str(sink_enabled)},{common_args}"
    return f"pretrained={model_path},conv_template=llada,model_name=llava_llada,{sink_args}"


def run_once(cmd: Sequence[str], env: dict, model_args: str, suffix: str) -> None:
    print(f"RUN_SUFFIX={suffix}")
    print("CMD:", " ".join(cmd))
    print(f"TASKS={env.get('TASKS', '')}")
    print(f"MODEL_ARGS={model_args}")
    print(f"GEN_ARGS={env.get('GEN_ARGS', '')}")
    print(f"LLAVA_OVERWRITE_IMAGE_ASPECT={env.get('LLAVA_OVERWRITE_IMAGE_ASPECT', '')}")
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    eval_script = resolve_eval_script(args.eval_script)
    sink_enabled = _as_bool(args.prefill_sink_enabled)

    os.makedirs(args.out_base, exist_ok=True)

    env = os.environ.copy()
    env["LLADA_VISION_ENCODER"] = args.llada_vision_encoder
    env["LLAVA_OVERWRITE_IMAGE_ASPECT"] = args.llava_overwrite_image_aspect
    env["NUM_PROCESSES"] = str(args.num_processes)
    env["NUM_MACHINES"] = str(args.num_machines)
    env["MACHINE_RANK"] = str(args.machine_rank)
    env["MAIN_PROCESS_PORT"] = str(args.main_process_port)
    env["TASKS"] = args.tasks
    env["GEN_ARGS"] = args.gen_args
    env["OUT_BASE"] = args.out_base
    env["PREFILL_SINK_COMMON_ARGS"] = args.prefill_sink_common_args
    env["PREFILL_SINK_ENABLED"] = str(sink_enabled)
    env["MODEL_PATH"] = args.model_path

    cmd, model_args, suffix = build_command(
        accelerate_binary=args.accelerate_binary,
        eval_script=eval_script,
        model_path=args.model_path,
        tasks=args.tasks,
        gen_args=args.gen_args,
        out_base=args.out_base,
        sink_enabled=sink_enabled,
        common_args=args.prefill_sink_common_args,
        num_processes=args.num_processes,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        main_process_port=args.main_process_port,
        extra_args=args.extra_args,
    )
    run_once(cmd, env, model_args, suffix)
    print(f"Done. Results under: {Path(args.out_base) / ('prefill_sink' if sink_enabled else 'baseline')}")


if __name__ == "__main__":
    main()
