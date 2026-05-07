#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


LABELS = ("TP", "FP", "TN", "FN")

BITS_TO_LABEL = {
    (True, True): "TP",
    (True, False): "FP",
    (False, True): "TN",
    (False, False): "FN",
}
LABEL_TO_BITS = {v: k for k, v in BITS_TO_LABEL.items()}


def parse_label_from_assistant(text: str) -> tuple[bool, bool, str] | None:
    g_m = re.search(r"Grounded:\s*(Yes|No)", text)
    c_m = re.search(r"Correct:\s*(Yes|No)", text)
    e_m = re.search(r"Evaluation:\s*(TP|FP|TN|FN)", text)
    if not (g_m and c_m and e_m):
        return None
    grounded = g_m.group(1) == "Yes"
    correct = c_m.group(1) == "Yes"
    evaluation = e_m.group(1)
    return grounded, correct, evaluation


def replace_header(text: str, grounded: bool, correct: bool, evaluation: str) -> str:
    text = re.sub(
        r"Grounded:\s*(Yes|No)",
        f"Grounded: {'Yes' if grounded else 'No'}",
        text,
        count=1,
    )
    text = re.sub(
        r"Correct:\s*(Yes|No)",
        f"Correct: {'Yes' if correct else 'No'}",
        text,
        count=1,
    )
    text = re.sub(
        r"Evaluation:\s*(TP|FP|TN|FN)",
        f"Evaluation: {evaluation}",
        text,
        count=1,
    )
    return text


def sample_rejected_label(
    gold_label: str,
    rng: random.Random,
    p_flip_correct: float = 0.5,
    p_flip_grounded: float = 0.3,
) -> str:
    g, c = LABEL_TO_BITS[gold_label]
    r = rng.random()
    if r < p_flip_correct:
        return BITS_TO_LABEL[(g, not c)]
    if r < p_flip_correct + p_flip_grounded:
        return BITS_TO_LABEL[(not g, c)]
    return BITS_TO_LABEL[(not g, not c)]


def build_pair(
    messages: list[dict],
    rng: random.Random,
    p_flip_correct: float,
    p_flip_grounded: float,
) -> dict | None:
    asst_msgs = [m for m in messages if m["role"] == "assistant"]
    if len(asst_msgs) != 1:
        return None
    asst = asst_msgs[0]["content"]
    parsed = parse_label_from_assistant(asst)
    if parsed is None:
        return None
    gold_grounded, gold_correct, gold_eval = parsed

    rejected_eval = sample_rejected_label(gold_eval, rng, p_flip_correct, p_flip_grounded)
    rej_grounded, rej_correct = LABEL_TO_BITS[rejected_eval]
    rejected_text = replace_header(asst, rej_grounded, rej_correct, rejected_eval)

    if rejected_text == asst:
        return None

    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    return {
        "prompt": prompt_msgs,
        "chosen": [{"role": "assistant", "content": asst}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "_meta": {
            "gold_label": gold_eval,
            "rejected_label": rejected_eval,
            "flip_correct": gold_correct != rej_correct,
            "flip_grounded": gold_grounded != rej_grounded,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", default="training_qwen.jsonl")
    p.add_argument("--output_path", default="dpo_pairs.jsonl")
    p.add_argument("--p_flip_correct", type=float, default=0.5)
    p.add_argument("--p_flip_grounded", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_meta", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.p_flip_correct < 0 or args.p_flip_grounded < 0:
        sys.exit("Probabilities must be non-negative.")
    if args.p_flip_correct + args.p_flip_grounded > 1.0 + 1e-9:
        sys.exit("p_flip_correct + p_flip_grounded must be <= 1.")

    rng = random.Random(args.seed)
    n_in = n_out = n_skipped = 0
    label_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    flip_kind_counts: Counter[str] = Counter()

    src = Path(args.input_path)
    dst = Path(args.output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            n_in += 1
            obj = json.loads(line)
            pair = build_pair(
                obj["messages"], rng,
                args.p_flip_correct, args.p_flip_grounded,
            )
            if pair is None:
                n_skipped += 1
                continue

            meta = pair.pop("_meta")
            label_counts[meta["gold_label"]] += 1
            rejected_counts[meta["rejected_label"]] += 1
            kind = (
                "flip_both" if (meta["flip_correct"] and meta["flip_grounded"])
                else "flip_correct" if meta["flip_correct"]
                else "flip_grounded"
            )
            flip_kind_counts[kind] += 1

            if args.keep_meta:
                pair["_meta"] = meta
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Read:    {n_in:,} examples from {src}")
    print(f"Wrote:   {n_out:,} pairs to {dst}")
    print(f"Skipped: {n_skipped:,} (header parse failed or single-asst missing)")
    print(f"Gold-label distribution:     {dict(label_counts)}")
    print(f"Rejected-label distribution: {dict(rejected_counts)}")
    print(f"Flip-kind distribution:      {dict(flip_kind_counts)}")


if __name__ == "__main__":
    main()
