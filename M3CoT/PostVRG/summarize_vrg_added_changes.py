#!/usr/bin/env python
import argparse
import csv
import json
import re
from pathlib import Path


RUNS = [
    {
        "key": "proposal",
        "label": "Proposal + VRG",
        "base_records": Path("M3CoT/PostVRG/outputs/main_proposal_postmask_seed42_n999999/records.jsonl"),
        "vrg_records": Path(
            "M3CoT/PostVRG/outputs/main_postvrg_alpha0p5_noise500_fixed32_refill2_seed42_n999999/records.jsonl"
        ),
        "trace_chunks": Path("M3CoT/PostVRG/outputs/main_postvrg_refill_logits_trace_full/chunks"),
    },
    {
        "key": "visual",
        "label": "Visual + VRG",
        "base_records": Path(
            "M3CoT/PostMaSK/outputs/postmask_visualgain_vcdnoise500_fixed32_refill2_seed42_n999999/records.jsonl"
        ),
        "vrg_records": Path(
            "M3CoT/PostMaSK/outputs/postmask_visualgain_vcdrefill_k4_alpha0p5_noise500_fixed32_refill2_seed42_n999999/records.jsonl"
        ),
        "trace_chunks": Path("M3CoT/PostVRG/outputs/visual_postmask_refill_logits_trace_full/chunks"),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize what changes after adding VRG.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("M3CoT/PostVRG/outputs/vrg_added_change_summary"),
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def step_records(row):
    return {
        int(record.get("step")): [int(pos) for pos in record.get("refilled_answer_positions") or []]
        for record in row.get("postmask_records") or []
    }


def pct(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarize_run(run):
    base_by_id = {row["id"]: row for row in load_jsonl(run["base_records"])}
    vrg_rows = load_jsonl(run["vrg_records"])

    final_changed = 0
    improved = 0
    worsened = 0
    both_correct = 0
    both_wrong = 0
    order_changed_samples = 0
    order_changed_steps = 0
    total_steps = 0

    for vrg in vrg_rows:
        base = base_by_id[vrg["id"]]
        final_changed += int(normalize_text(base.get("final_text")) != normalize_text(vrg.get("final_text")))
        base_correct = bool(base.get("final_correct"))
        vrg_correct = bool(vrg.get("final_correct"))
        if (not base_correct) and vrg_correct:
            improved += 1
        elif base_correct and (not vrg_correct):
            worsened += 1
        elif base_correct and vrg_correct:
            both_correct += 1
        else:
            both_wrong += 1

        base_steps = step_records(base)
        vrg_steps = step_records(vrg)
        sample_order_changed = False
        for step in sorted(set(base_steps) & set(vrg_steps)):
            total_steps += 1
            if base_steps[step] != vrg_steps[step]:
                sample_order_changed = True
                order_changed_steps += 1
        order_changed_samples += int(sample_order_changed)

    selected_total = 0
    selected_changed = 0
    active_total = 0
    active_changed = 0
    selected_switch_samples = 0
    selected_rank_changed = 0
    guided_only_selected = 0
    cond_only_selected = 0
    selected_events = 0
    replay_match = 0
    trace_samples = 0

    for chunk_dir in sorted(path for path in run["trace_chunks"].glob("chunk_*") if path.is_dir()):
        summary_path = chunk_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            trace_samples += int(summary.get("num_samples", 0))
            replay_match += int(summary.get("replay_match_count", 0))
            selected_total += int(summary.get("selected_total", 0))
            selected_changed += int(summary.get("selected_changed", 0))
            active_total += int(summary.get("active_total", 0))
            active_changed += int(summary.get("active_changed", 0))
            selected_switch_samples += sum(
                1 for item in summary.get("sample_summaries") or [] if int(item.get("selected_changed", 0)) > 0
            )
        events_path = chunk_dir / "events.jsonl"
        if events_path.exists():
            with events_path.open(encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("scope") != "selected":
                        continue
                    selected_events += 1
                    by_cond = bool(event.get("selected_by_cond_topk"))
                    by_guided = bool(event.get("selected_by_guided_topk"))
                    guided_only_selected += int((not by_cond) and by_guided)
                    cond_only_selected += int(by_cond and (not by_guided))
                    selected_rank_changed += int(int(event.get("rank_delta", 0)) != 0 or by_cond != by_guided)

    samples = len(vrg_rows)
    return {
        "run": run["label"],
        "samples": samples,
        "final_text_changed": final_changed,
        "final_text_changed_rate": pct(final_changed, samples),
        "improved": improved,
        "worsened": worsened,
        "net_correct": improved - worsened,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "refill_order_changed_samples": order_changed_samples,
        "refill_order_changed_sample_rate": pct(order_changed_samples, samples),
        "refill_order_changed_steps": order_changed_steps,
        "total_refill_steps": total_steps,
        "refill_order_changed_step_rate": pct(order_changed_steps, total_steps),
        "selected_top1_changed": selected_changed,
        "selected_total": selected_total,
        "selected_top1_changed_rate": pct(selected_changed, selected_total),
        "selected_top1_changed_samples": selected_switch_samples,
        "selected_top1_changed_sample_rate": pct(selected_switch_samples, trace_samples),
        "guided_only_selected": guided_only_selected,
        "guided_only_selected_rate": pct(guided_only_selected, selected_events),
        "cond_only_selected": cond_only_selected,
        "selected_rank_changed": selected_rank_changed,
        "selected_rank_changed_rate": pct(selected_rank_changed, selected_events),
        "active_top1_changed": active_changed,
        "active_total": active_total,
        "active_top1_changed_rate": pct(active_changed, active_total),
        "replay_match": replay_match,
        "trace_samples": trace_samples,
    }


def write_csv(path, rows):
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100.0 * value:.2f}%"


def write_markdown(path, rows):
    proposal_only = [row for row in rows if row["run"] == "Proposal + VRG"]
    metrics = [
        ("Final text changed", "final_text_changed", "final_text_changed_rate"),
        ("Improved / worsened / net", None, None),
        ("Refill order changed steps", "refill_order_changed_steps", "refill_order_changed_step_rate"),
        ("Selected top1 changed", "selected_top1_changed", "selected_top1_changed_rate"),
        ("Guided-only selected", "guided_only_selected", "guided_only_selected_rate"),
        ("Selected rank changed", "selected_rank_changed", "selected_rank_changed_rate"),
    ]
    lines = [
        "# Changes After Adding VRG",
        "",
        "| key metric | Proposal + VRG |",
        "|---|---:|",
    ]
    proposal = proposal_only[0]
    for label, count_key, rate_key in metrics:
        if label == "Improved / worsened / net":
            left = f"{proposal['improved']} / {proposal['worsened']} / {proposal['net_correct']:+d}"
        else:
            left = f"{proposal[count_key]} ({percent(proposal[rate_key])})"
        lines.append(f"| {label} | {left} |")
    lines.extend(
        [
            "",
            "Interpretation: Proposal+VRG mainly changes the refill path. The key evidence is the large refill-order change rate plus non-trivial guided-only selected and selected-rank changes; selected top1 flips exist but are rare, so the gain is not mainly direct answer-token replacement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_key_csv(path, rows):
    proposal = next(row for row in rows if row["run"] == "Proposal + VRG")
    key_rows = [
        ("Final text changed", proposal["final_text_changed"], proposal["final_text_changed_rate"]),
        ("Improved", proposal["improved"], proposal["improved"] / proposal["samples"]),
        ("Worsened", proposal["worsened"], proposal["worsened"] / proposal["samples"]),
        ("Net correct", proposal["net_correct"], proposal["net_correct"] / proposal["samples"]),
        (
            "Refill order changed steps",
            proposal["refill_order_changed_steps"],
            proposal["refill_order_changed_step_rate"],
        ),
        ("Selected top1 changed", proposal["selected_top1_changed"], proposal["selected_top1_changed_rate"]),
        ("Guided-only selected", proposal["guided_only_selected"], proposal["guided_only_selected_rate"]),
        ("Selected rank changed", proposal["selected_rank_changed"], proposal["selected_rank_changed_rate"]),
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["metric", "count", "rate"])
        writer.writeheader()
        for metric, count, rate in key_rows:
            writer.writerow({"metric": metric, "count": count, "rate": rate})


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(run) for run in RUNS]
    (args.output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "summary.md", rows)
    write_key_csv(args.output_dir / "proposal_key_summary.csv", rows)
    for row in rows:
        print(
            f"{row['run']}: final_changed={row['final_text_changed']} "
            f"selected_top1={row['selected_top1_changed']} guided_only={row['guided_only_selected']} "
            f"order_steps={row['refill_order_changed_steps']}"
        )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
