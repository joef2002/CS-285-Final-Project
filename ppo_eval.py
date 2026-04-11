#!/usr/bin/env python3
"""Evaluate evaluator checkpoints on test_qwen.jsonl with parse-aware metrics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


LABELS = ("TP", "TN", "FP", "FN")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter_path", type=str, default="final_adapter/final_adapter")
    p.add_argument("--test_path", type=str, default="test_qwen.jsonl")
    p.add_argument("--output_path", type=str, default="ppo_eval_results.jsonl")
    p.add_argument("--max_prompt_tokens", type=int, default=16384)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--strict_json_only", action="store_true")
    return p.parse_args()


def extract_first_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*?\}", text, flags=re.S)
    return m.group(0) if m else None


def parse_prediction(text: str, strict_json_only: bool) -> tuple[str | None, dict[str, Any] | None]:
    blob = extract_first_json_object(text)
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


def main() -> None:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    test_data = [json.loads(line) for line in Path(args.test_path).read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(test_data):,} test examples")

    counts = Counter()
    conf = Counter()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for i, example in enumerate(tqdm(test_data, desc="Inference")):
            messages = example["messages"]
            expected = example.get("expected", {})
            expected_label = str(expected.get("evaluation", "")).upper()

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            pred_label, parsed = parse_prediction(response_text, strict_json_only=args.strict_json_only)
            pred_for_conf = pred_label if pred_label in LABELS else "PARSE_FAIL"
            counts["n"] += 1
            if pred_for_conf != "PARSE_FAIL":
                counts["parse_ok"] += 1
            if pred_for_conf in LABELS and pred_for_conf == expected_label:
                counts["correct"] += 1
            conf[(expected_label, pred_for_conf)] += 1

            out.write(
                json.dumps(
                    {
                        "index": i,
                        "expected": expected,
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
    print("\n=== Metrics ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro-F1:   {mf1:.4f}")
    print(f"Parse rate: {parse_rate:.4f}")
    for lbl in LABELS:
        tp = conf[(lbl, lbl)]
        fp = sum(conf[(g, lbl)] for g in LABELS if g != lbl)
        fn = sum(conf[(lbl, p)] for p in LABELS if p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"{lbl}: precision={prec:.4f} recall={rec:.4f}")
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
