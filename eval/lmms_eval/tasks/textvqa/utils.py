import datetime
import json
import os
import pathlib
import re
import statistics

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


TEXTVQA_SHORT_PROMPT = "Answer the question using a single word or phrase."
TEXTVQA_REASONING_PROMPT = (
    "Please reason step by step, and answer the question using a single word or phrase "
    "in the format of Answer: <answer>."
)


def _clean_extracted_answer(text):
    text = text.strip().strip("\"'`*")
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".,;:!?")
    return text


def _short_candidate_or_none(text, max_words=8):
    candidate = _clean_extracted_answer(text)
    if not candidate:
        return None
    if len(candidate.split()) > max_words:
        return None
    return candidate


def _extract_final_answer(text):
    text = text.replace("<|endoftext|>", "").replace("<|eot_id|>", "").strip()

    for pattern in (r"Answer\s*:\s*(.+)", r"Final answer\s*:\s*(.+)"):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            candidate = _short_candidate_or_none(matches[-1].splitlines()[0])
            if candidate is not None:
                return candidate

    boxed_matches = re.findall(r"\\boxed\s*{([^{}]+)}", text)
    if boxed_matches:
        candidate = _short_candidate_or_none(boxed_matches[-1])
        if candidate is not None:
            return candidate

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        last_line = nonempty_lines[-1]
        candidate = _short_candidate_or_none(last_line)
        if candidate is not None and not re.fullmatch(r"(therefore|thus|so)[,:]?", candidate, flags=re.IGNORECASE):
            return candidate

    quoted_matches = re.findall(r'"([^"\n]{1,80})"|\'([^\'\n]{1,80})\'', text)
    if quoted_matches:
        flat_matches = [a or b for a, b in quoted_matches]
        candidate = _short_candidate_or_none(flat_matches[-1])
        if candidate is not None:
            return candidate

    sentence_patterns = (
        r"(?:therefore|thus|so)[^.\n]*?\b(?:is|are|was|were)\s+([^.\n]+)",
        r"\b(?:answer|number|brand|word|time|title|name|value|type|state|color|event|measurement)\b[^.\n]*?\b(?:is|are|was|were)\s+([^.\n]+)",
    )
    for pattern in sentence_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            candidate = _short_candidate_or_none(matches[-1])
            if candidate is not None:
                return candidate

    tail = text.splitlines()[-1] if text.splitlines() else text
    return _clean_extracted_answer(tail)


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(_extract_final_answer(result[0]))
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }


def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    ocr_ref = ""
    env_prompt_mode = os.environ.get("TEXTVQA_PROMPT_MODE", "").strip().lower()
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        if "prompt_mode" in lmms_eval_specific_kwargs:
            prompt_mode = lmms_eval_specific_kwargs["prompt_mode"]
            if prompt_mode == "reasoning":
                post_prompt = f"\n{TEXTVQA_REASONING_PROMPT}"
            elif prompt_mode == "short":
                post_prompt = f"\n{TEXTVQA_SHORT_PROMPT}"
        if "ocr" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["ocr"]:
            ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"
    if env_prompt_mode == "reasoning":
        post_prompt = f"\n{TEXTVQA_REASONING_PROMPT}"
    elif env_prompt_mode == "short":
        post_prompt = f"\n{TEXTVQA_SHORT_PROMPT}"
    return f"{pre_prompt}{doc['question'].capitalize()}{ocr_ref}{post_prompt}"


def textvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    # print(f"Submission file saved to {path}")
    eval_logger.info(f"Submission file saved to {path}")
