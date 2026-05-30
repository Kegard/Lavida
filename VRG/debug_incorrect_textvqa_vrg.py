import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VRG trace visualization for samples listed in incorrect_textvqa_val.json."
    )
    parser.add_argument("--incorrect-json", default="VRG/incorrect_textvqa_val.json")
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0, help="Start index inside the incorrect samples list.")
    parser.add_argument("--limit", type=int, default=5, help="Number of incorrect samples to debug.")
    parser.add_argument("--output-dir", default="VRG/outputs/incorrect_textvqa_vrg")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])

    parser.add_argument("--vrg-alpha-start", type=float, default=0.0)
    parser.add_argument("--vrg-alpha-end", type=float, default=1.0)
    parser.add_argument("--vrg-alpha-schedule", default="linear", choices=["linear", "cosine", "power"])
    parser.add_argument("--vrg-alpha-power", type=float, default=2.0)
    parser.add_argument("--vrg-null-visual-mode", default="zeros", choices=["zeros", "mask_token"])
    parser.add_argument("--trace-topk", type=int, default=10)
    parser.add_argument("--trace-max-positions", type=int, default=4)
    return parser.parse_args()


def load_error_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported incorrect JSON format: {path}")


def build_sample_from_error_record(args, dataset, record):
    dataset_index = int(record["doc_id"])
    doc = dataset[dataset_index]
    question = record.get("input")
    if not question:
        question = f"{record['question'].capitalize()}\nAnswer the question using a single word or phrase."

    return {
        "image": doc["image"].convert("RGB"),
        "image_source": f"{args.dataset_path}:{args.split}:{dataset_index}",
        "question": question,
        "dataset_meta": {
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "dataset_index": dataset_index,
            "doc_id": record.get("doc_id"),
            "question_id": record.get("question_id"),
            "original_question": record.get("question"),
            "eval_prediction": record.get("prediction"),
            "submission_answer": record.get("submission_answer"),
            "answers": record.get("answers"),
            "ocr_tokens": record.get("ocr_tokens"),
            "exact_match": record.get("exact_match"),
        },
    }


def save_trace_plots(trace_json_path, output_dir, dpi):
    from VRG.visualize_vrg_logits import (
        save_alpha_plot,
        save_delta_plot,
        save_position_heatmap,
        save_summary,
        save_topk_bar_plot,
    )

    trace_payload = json.loads(Path(trace_json_path).read_text(encoding="utf-8"))
    trace_records = trace_payload["trace_records"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_alpha_plot(trace_records, output_dir / "alpha_over_steps.png", dpi)
    save_delta_plot(trace_records, output_dir / "delta_over_steps.png", dpi)
    save_position_heatmap(trace_records, output_dir / "position_delta_heatmap.png", dpi)
    save_topk_bar_plot(trace_records, output_dir / "topk_delta_tokens.png", dpi)
    save_summary(trace_payload, output_dir / "summary.json")


def main():
    args = parse_args()
    from VRG.debug_timestep_vrg import (
        get_torch_dtype,
        load_dataset_split,
        maybe_disable_torch_compile,
        run_one_sample,
    )
    from llava.model.builder import load_pretrained_model

    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    args.save_trace = True
    args.output = str(Path(args.output_dir) / "placeholder.json")

    records = load_error_records(args.incorrect_json)
    if args.start_index < 0 or args.start_index >= len(records):
        raise ValueError(f"--start-index must be within [0, {len(records) - 1}].")

    selected = records[args.start_index : args.start_index + args.limit]
    dataset = load_dataset_split(args.dataset_path, args.dataset_name, args.split)

    restore_compile = maybe_disable_torch_compile()
    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        args.model_name,
        device_map=f"{args.device}:0" if args.device.startswith("cuda") else args.device,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_jsonl = output_dir / "incorrect_trace_summary.jsonl"

    summaries = []
    for list_index, record in enumerate(selected, start=args.start_index):
        sample = build_sample_from_error_record(args, dataset, record)
        stem = f"incorrect_{list_index:06d}_doc_{int(record['doc_id']):06d}"
        sample_dir = output_dir / stem
        output_path = sample_dir / "generation.json"
        trace_output_path = sample_dir / "trace.json"
        trace_pt_output_path = sample_dir / "trace.pt"

        print(
            "Running incorrect sample "
            f"list_index={list_index}, doc_id={record.get('doc_id')}, "
            f"question_id={record.get('question_id')}: {sample['question'][:120]}"
        )
        payload = run_one_sample(
            args=args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            sample=sample,
            output_path=output_path,
            trace_output_path=trace_output_path,
            trace_pt_output_path=trace_pt_output_path,
        )

        plots_dir = sample_dir / "plots"
        if not args.skip_visualize:
            save_trace_plots(trace_output_path, plots_dir, args.dpi)

        summaries.append(
            {
                "list_index": list_index,
                "doc_id": record.get("doc_id"),
                "question_id": record.get("question_id"),
                "question": record.get("question"),
                "eval_prediction": record.get("prediction"),
                "rerun_final_text": payload["final_text"],
                "answers": record.get("answers"),
                "exact_match": record.get("exact_match"),
                "generation_json": str(output_path),
                "trace_json": str(trace_output_path),
                "trace_pt": str(trace_pt_output_path),
                "plots_dir": str(plots_dir) if not args.skip_visualize else None,
            }
        )

    with summary_jsonl.open("w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Saved incorrect trace summary to {summary_jsonl}")
    restore_compile()


if __name__ == "__main__":
    main()
