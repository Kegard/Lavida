#!/usr/bin/env python
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from M3CoT.PostMaSK.run_m3cot_postmask import main


def cli_value(flag, default):
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def add_default(flag, value):
    if value is None:
        return
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    alpha = cli_value("--vcd-refill-alpha", "1.0")
    calibration = cli_value("--refill-vrg-calibration", "none")
    confidence_threshold = cli_value("--refill-vrg-confidence-threshold", "0.9")
    confidence_gate_tau = cli_value("--refill-vrg-confidence-gate-tau", None)
    guidance_steps = cli_value("--refill-guidance-steps", None)
    noise_step = cli_value("--vcd-noise-step", "500")
    weak_mode = cli_value("--refill-weak-visual-mode", "diffusion_noise")
    sample_seed = cli_value("--sample-seed", "42")
    limit = cli_value("--limit", "400")
    remask_selection = cli_value("--remask-selection", "proposal_confidence")
    position_boost_lambda = cli_value("--position-boost-lambda", "0.0")
    position_boost_q2 = cli_value("--position-boost-q2", "0.0")
    position_boost_q3 = cli_value("--position-boost-q3", "0.5")
    position_boost_q4 = cli_value("--position-boost-q4", "1.0")
    weak_tag = "nullvisual" if weak_mode == "null_visual" else f"noise{noise_step}"
    if calibration == "none":
        calib_tag = f"alpha{safe_name(alpha)}"
    elif calibration == "soft_confidence":
        calib_tag = f"softconf_alpha{safe_name(alpha)}"
    elif calibration == "hard_confidence":
        calib_tag = f"hardconf_tau{safe_name(confidence_threshold)}_alpha{safe_name(alpha)}"
    else:
        calib_tag = f"{calibration}_alpha{safe_name(alpha)}"
    step_tag = "" if guidance_steps is None else f"_k{safe_name(guidance_steps)}"
    gate_tag = "" if confidence_gate_tau is None else f"_gate{safe_name(confidence_gate_tau)}"
    selector_tag = "proposalconf"
    if remask_selection == "proposal_confidence_position_boost":
        selector_tag = (
            "proposalconf_positionboost"
            f"_l{safe_name(position_boost_lambda)}"
            f"_q2{safe_name(position_boost_q2)}"
            f"_q3{safe_name(position_boost_q3)}"
            f"_q4{safe_name(position_boost_q4)}"
        )
    default_output = (
        "M3CoT/PostVRG/outputs/"
        f"postvrg_{selector_tag}_vcdrefill_{calib_tag}{step_tag}{gate_tag}_"
        f"{weak_tag}_seed{sample_seed}_n{limit}"
    )

    defaults = {
        "--prompt": "cot",
        "--max-new-tokens": 64,
        "--block-length": 64,
        "--step-ratio": 0.5,
        "--limit": limit,
        "--sample-mode": "random",
        "--sample-seed": sample_seed,
        "--remask-selection": remask_selection,
        "--draft-steps": 16,
        "--postmask-steps": 16,
        "--remask-per-step": 4,
        "--postmask-mode": "fixed_set",
        "--fixed-set-size": 32,
        "--fixed-refill-per-step": 2,
        "--refill-guidance": "vcd",
        "--refill-weak-visual-mode": weak_mode,
        "--vcd-refill-alpha": alpha,
        "--refill-vrg-calibration": calibration,
        "--refill-vrg-confidence-threshold": confidence_threshold,
        "--refill-vrg-confidence-gate-tau": confidence_gate_tau,
        "--refill-guidance-steps": guidance_steps,
        "--vcd-noise-step": noise_step,
        "--vcd-noise-seed": 42,
        "--position-boost-lambda": position_boost_lambda,
        "--position-boost-q2": position_boost_q2,
        "--position-boost-q3": position_boost_q3,
        "--position-boost-q4": position_boost_q4,
        "--output-dir": default_output,
    }
    for flag, value in defaults.items():
        add_default(flag, value)

    main()
