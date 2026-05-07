#!/usr/bin/env python3
"""Evaluate an existing prediction JSONL file with parse-aware metrics.

Expected by default for each JSONL row:
- expected.evaluation: gold label in {TP, TN, FP, FN}
- model_response: raw model text to parse
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("TP", "TN", "FP", "FN")


def strip_think_prefix(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_first_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*?\}", text, flags=re.S)
    return m.group(0) if m else None


def parse_prediction(text: str, strict_json_only: bool) -> tuple[str | None, dict[str, Any] | None]:
    clean_text = strip_think_prefix(text)
    blob = extract_first_json_object(clean_text)
    parsed: dict[str, Any] | None = None
    if blob is not None:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                parsed = obj
        except json.JSONDecodeError:
            parsed = None

    if parsed is not None:
        label = str(parsed.get("evaluation", "")).upper()
        if label in LABELS:
            return label, parsed

    if strict_json_only:
        return None, parsed

    m = re.search(r"\b(TP|TN|FP|FN)\b", text)
    return (m.group(1) if m else None), parsed


def macro_f1(conf: Counter[tuple[str, str]]) -> float:
    def f1(lbl: str) -> float:
        tp = conf[(lbl, lbl)]
        fp = sum(conf[(g, lbl)] for g in LABELS if g != lbl)
        fn = sum(conf[(lbl, p)] for p in LABELS if p != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return sum(f1(lbl) for lbl in LABELS) / len(LABELS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions_path", type=str, default="sft_test_results.jsonl")
    p.add_argument("--output_path", type=str, default="jsonl_eval_results.jsonl")
    p.add_argument("--strict_json_only", action="store_true")
    p.add_argument("--expected_field", type=str, default="expected")
    p.add_argument("--expected_label_field", type=str, default="evaluation")
    p.add_argument("--response_field", type=str, default="model_response")
    p.add_argument(
        "--label_weights",
        type=str,
        default="TP=0.89,TN=0.068,FP=0.038,FN=0.004",
        help=(
            "Comma-separated label weights, e.g. 'TP=0.89,TN=0.068,FP=0.038,FN=0.004'. "
            "Defaults to the screenshot distribution and will be normalized to sum to 1."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = [json.loads(line) for line in Path(args.predictions_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(rows):,} rows from {args.predictions_path}")

    counts = Counter()
    conf = Counter()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            expected_obj = row.get(args.expected_field, {})
            expected_label = str(expected_obj.get(args.expected_label_field, "")).upper()
            response_text = str(row.get(args.response_field, ""))

            pred_label, parsed = parse_prediction(response_text, strict_json_only=args.strict_json_only)
            pred_for_conf = pred_label if pred_label in LABELS else "PARSE_FAIL"

            counts["n"] += 1
            if expected_label == "TP":
                counts["tp_gold"] += 1
            elif expected_label in LABELS:
                counts["non_tp_gold"] += 1

            if pred_for_conf != "PARSE_FAIL":
                counts["parse_ok"] += 1
            if pred_for_conf in LABELS and pred_for_conf == expected_label:
                counts["correct"] += 1
            if expected_label == "TP" and pred_for_conf == "TP":
                counts["tp_caught"] += 1
            if expected_label in LABELS and expected_label != "TP" and pred_for_conf in LABELS and pred_for_conf != "TP":
                counts["non_tp_caught"] += 1
            conf[(expected_label, pred_for_conf)] += 1

            out.write(
                json.dumps(
                    {
                        "index": row.get("index", i),
                        "expected": expected_obj,
                        "predicted_label": pred_label,
                        "parsed": parsed,
                        "model_response": response_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    acc = counts["correct"] / max(1, counts["n"])
    parse_rate = counts["parse_ok"] / max(1, counts["n"])
    mf1 = macro_f1(conf)
    # compute per-label gold counts and per-label correct (catch) rates
    gold_counts: dict[str, int] = {}
    catch_rates: dict[str, float] = {}
    for lbl in LABELS:
        gold = sum(conf[(lbl, p)] for p in LABELS) + conf[(lbl, "PARSE_FAIL")]
        gold_counts[lbl] = gold
        caught = conf[(lbl, lbl)]
        catch_rates[lbl] = (caught / gold) if gold else 0.0

    # parse label weights from args (format: TP=0.89,TN=0.068,...). Default: uniform
    if args.label_weights:
        parts = [p.strip() for p in args.label_weights.split(",") if p.strip()]
        weights: dict[str, float] = {}
        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip().upper()
            try:
                weights[k] = float(v)
            except ValueError:
                weights[k] = 0.0
        # fill missing labels with 0
        for lbl in LABELS:
            weights.setdefault(lbl, 0.0)
        total_w = sum(weights.values())
        if total_w <= 0:
            # fallback to uniform
            weights = {lbl: 1.0 / len(LABELS) for lbl in LABELS}
        else:
            weights = {lbl: weights[lbl] / total_w for lbl in LABELS}
    else:
        weights = {lbl: 1.0 / len(LABELS) for lbl in LABELS}

    # weighted accuracy = sum_{label} weight[label] * catch_rate[label]
    weighted_acc = sum(weights[lbl] * catch_rates[lbl] for lbl in LABELS)

    # non-TP catch rate: proportion of non-TP gold labels that were correctly
    # predicted as non-TP (uses counts tracked during evaluation)
    non_tp_catch_rate = counts.get("non_tp_caught", 0) / max(1, counts.get("non_tp_gold", 0))

    # parsed-only per-label gold counts and catch rates (ignore PARSE_FAIL)
    parsed_gold_counts: dict[str, int] = {}
    parsed_catch_rates: dict[str, float] = {}
    for lbl in LABELS:
        gold_parsed = sum(conf[(lbl, p)] for p in LABELS)
        parsed_gold_counts[lbl] = gold_parsed
        caught = conf[(lbl, lbl)]
        parsed_catch_rates[lbl] = (caught / gold_parsed) if gold_parsed else 0.0

    parsed_weighted_acc = sum(weights[lbl] * parsed_catch_rates[lbl] for lbl in LABELS)

    # parsed non-TP catch rate: only considers parsed non-TP gold labels
    parsed_non_tp_gold = sum(parsed_gold_counts[lbl] for lbl in LABELS if lbl != "TP")
    parsed_non_tp_caught = sum(conf[(lbl, lbl)] for lbl in LABELS if lbl != "TP")
    parsed_non_tp_catch_rate = parsed_non_tp_caught / max(1, parsed_non_tp_gold)

    print("\n=== Metrics ===")
    print(f"Weighted Accuracy: {weighted_acc:.4f}")
    print(f"Non-TP Catch Rate: {non_tp_catch_rate:.4f}")
    print(f"Parsed Weighted Accuracy: {parsed_weighted_acc:.4f}")
    print(f"Parsed Non-TP Catch Rate: {parsed_non_tp_catch_rate:.4f}")
    print("Per-label weights and catch rates:")
    for lbl in LABELS:
        print(f"  {lbl}: weight={weights[lbl]:.4f} catch_rate={catch_rates[lbl]:.4f} (gold={gold_counts[lbl]})")
    print(f"Exact Accuracy:    {acc:.4f}")
    print(f"Macro-F1:   {mf1:.4f}")
    print(f"Parse rate: {parse_rate:.4f}")
    for lbl in LABELS:
        tp = conf[(lbl, lbl)]
        fp = sum(conf[(g, lbl)] for g in LABELS if g != lbl)
        fn = sum(conf[(lbl, p)] for p in LABELS if p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"{lbl}: precision={prec:.4f} recall={rec:.4f}")
    parse_fail = counts["n"] - counts["parse_ok"]
    print(f"Parse failures: {parse_fail}")
    print(f"Wrote normalized results to {output_path}")


if __name__ == "__main__":
    main()
