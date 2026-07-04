#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize selected refill VRG key metrics from trace chunks.")
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def iter_chunk_dirs(chunks_root):
    return sorted(path for path in chunks_root.glob("chunk_*") if path.is_dir())


def pct(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarize(chunks_root):
    selected_total = 0
    selected_top1_changed = 0
    active_total = 0
    active_top1_changed = 0
    guided_only_selected = 0
    cond_only_selected = 0
    selected_rank_changed = 0
    selected_events = 0
    num_samples = 0
    replay_match = 0

    chunk_dirs = iter_chunk_dirs(chunks_root)
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk directories found under {chunks_root}")

    for chunk_dir in chunk_dirs:
        summary_path = chunk_dir / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            num_samples += int(summary.get("num_samples", 0))
            replay_match += int(summary.get("replay_match_count", 0))
            selected_total += int(summary.get("selected_total", 0))
            selected_top1_changed += int(summary.get("selected_changed", 0))
            active_total += int(summary.get("active_total", 0))
            active_top1_changed += int(summary.get("active_changed", 0))

        events_path = chunk_dir / "events.jsonl"
        if not events_path.exists():
            continue
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

    return {
        "chunks_root": str(chunks_root),
        "num_chunks": len(chunk_dirs),
        "num_samples": num_samples,
        "replay_match": replay_match,
        "selected_total": selected_total,
        "selected_events": selected_events,
        "selected_top1_changed": selected_top1_changed,
        "selected_top1_changed_rate": pct(selected_top1_changed, selected_total),
        "guided_only_selected": guided_only_selected,
        "guided_only_selected_rate": pct(guided_only_selected, selected_events),
        "cond_only_selected": cond_only_selected,
        "cond_only_selected_rate": pct(cond_only_selected, selected_events),
        "selected_rank_changed": selected_rank_changed,
        "selected_rank_changed_rate": pct(selected_rank_changed, selected_events),
        "active_total": active_total,
        "active_top1_changed": active_top1_changed,
        "active_top1_changed_rate": pct(active_top1_changed, active_total),
    }


def main():
    args = parse_args()
    result = summarize(args.chunks_root)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
