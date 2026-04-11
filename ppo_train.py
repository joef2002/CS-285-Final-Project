#!/usr/bin/env python3
"""PPO fine-tuning for the reticular-chemistry evaluator from an SFT adapter."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an expert evaluator for reticular chemistry question-answer pairs. "
    "Given a manuscript context, a question, and an answer, decide whether the "
    "question is grounded in the context and whether the answer is correct. "
    "Return ONLY valid JSON with keys grounded (bool), correct (bool), "
    'evaluation ("TP"|"TN"|"FP"|"FN").'
)

LABELS = ("TP", "TN", "FP", "FN")
PAIR_TO_LABEL = {
    (True, True): "TP",
    (False, True): "TN",
    (True, False): "FP",
    (False, False): "FN",
}


@dataclass
class Example:
    doi: str
    prompt: str
    grounded: bool
    correct: bool
    evaluation: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Model + paths
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--sft_adapter_path", type=str, default="final_adapter/final_adapter")
    p.add_argument("--training_set_path", type=str, default="training_set.json")
    p.add_argument("--paper_root", type=str, default=".")
    p.add_argument("--output_dir", type=str, default="runs/ppo_from_sft")
    p.add_argument("--cache_prompts_path", type=str, default="")

    # Data
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_context_chars", type=int, default=12000)
    p.add_argument("--max_prompt_tokens", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.9)

    # PPO
    p.add_argument("--total_ppo_updates", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mini_batch_size", type=int, default=2)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--target_kl", type=float, default=0.05)
    p.add_argument("--init_kl_coef", type=float, default=0.02)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Reward
    p.add_argument("--reward_format_weight", type=float, default=0.3)
    p.add_argument("--reward_consistency_weight", type=float, default=0.2)
    p.add_argument("--reward_label_weight", type=float, default=0.5)
    p.add_argument("--class_balance_alpha", type=float, default=0.5)

    # Logging / eval
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_max_examples", type=int, default=256)

    return p.parse_args()


def read_text_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_candidate_path(raw_path: str, paper_root: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return paper_root / p


def load_context(ms: str, si: str, paper_root: Path, max_context_chars: int) -> str:
    chunks: list[str] = []
    if ms:
        ms_text = read_text_if_exists(normalize_candidate_path(ms, paper_root))
        if ms_text:
            chunks.append("<ms>\n" + ms_text + "\n</ms>")
    if si:
        si_parts = []
        for raw in si.splitlines():
            rp = raw.strip()
            if not rp:
                continue
            txt = read_text_if_exists(normalize_candidate_path(rp, paper_root))
            if txt:
                si_parts.append(txt)
        if si_parts:
            chunks.append("<si>\n" + "\n\n".join(si_parts) + "\n</si>")
    context = "\n\n".join(chunks).strip()
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    return context


def build_user_prompt(context: str, question: str, answer: str) -> str:
    return (
        f"<context>\n{context}\n</context>\n\n"
        f"<question>{question}</question>\n"
        f"<answer>{answer}</answer>\n\n"
        "Evaluate this question-answer pair."
    )


def convert_training_set_to_examples(
    training_set_path: Path,
    paper_root: Path,
    max_context_chars: int,
) -> list[Example]:
    raw = json.loads(training_set_path.read_text(encoding="utf-8"))
    examples: list[Example] = []
    skipped_missing_context = 0
    skipped_bad_label = 0
    for paper in tqdm(raw, desc="Building PPO examples"):
        doi = str(paper.get("doi", ""))
        context = load_context(
            ms=str(paper.get("ms", "")),
            si=str(paper.get("si", "")),
            paper_root=paper_root,
            max_context_chars=max_context_chars,
        )
        if not context:
            skipped_missing_context += 1
            continue
        for qa in paper.get("QA pairs", []):
            label = str(qa.get("evaluation", "")).upper()
            if label not in LABELS:
                skipped_bad_label += 1
                continue
            grounded = bool(qa.get("grounded", False))
            correct = bool(qa.get("correct", False))
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue
            examples.append(
                Example(
                    doi=doi,
                    prompt=build_user_prompt(context, question, answer),
                    grounded=grounded,
                    correct=correct,
                    evaluation=label,
                )
            )
    print(f"Built {len(examples):,} PPO examples")
    print(f"Skipped papers with no readable MS/SI context: {skipped_missing_context:,}")
    print(f"Skipped QA rows with invalid label: {skipped_bad_label:,}")
    return examples


def split_by_doi(examples: list[Example], val_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    dois = sorted({ex.doi for ex in examples})
    rnd = random.Random(seed)
    rnd.shuffle(dois)
    cut = max(1, int(len(dois) * (1 - val_ratio)))
    train_dois = set(dois[:cut])
    train = [ex for ex in examples if ex.doi in train_dois]
    val = [ex for ex in examples if ex.doi not in train_dois]
    return train, val


def class_weights(examples: list[Example], alpha: float) -> dict[str, float]:
    counts = Counter(ex.evaluation for ex in examples)
    max_count = max(counts.values())
    out = {}
    for lbl in LABELS:
        c = max(1, counts.get(lbl, 1))
        out[lbl] = float((max_count / c) ** alpha)
    return out


def extract_first_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*?\}", text, flags=re.S)
    return m.group(0) if m else None


def parse_prediction(response_text: str) -> dict[str, Any] | None:
    blob = extract_first_json_object(response_text)
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def reward_for_prediction(
    parsed: dict[str, Any] | None,
    gold: Example,
    weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    metrics = {
        "format": 0.0,
        "consistency": 0.0,
        "label": 0.0,
    }
    if parsed is None:
        total = -args.reward_label_weight
        return total, metrics

    grounded = parsed.get("grounded")
    correct = parsed.get("correct")
    evaluation = str(parsed.get("evaluation", "")).upper()
    has_valid_types = isinstance(grounded, bool) and isinstance(correct, bool)
    has_valid_label = evaluation in LABELS

    if has_valid_types and has_valid_label:
        metrics["format"] = 1.0

    if has_valid_types and has_valid_label:
        mapped = PAIR_TO_LABEL.get((grounded, correct))
        if mapped == evaluation:
            metrics["consistency"] = 1.0

    if has_valid_label and evaluation == gold.evaluation:
        metrics["label"] = weights[gold.evaluation]
    elif has_valid_label:
        metrics["label"] = -weights[gold.evaluation]
    else:
        metrics["label"] = -weights[gold.evaluation]

    total = (
        args.reward_format_weight * metrics["format"]
        + args.reward_consistency_weight * metrics["consistency"]
        + args.reward_label_weight * metrics["label"]
    )
    return total, metrics


def build_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_batch(examples: list[Example], batch_size: int, rnd: random.Random) -> list[Example]:
    if batch_size >= len(examples):
        return list(examples)
    return rnd.sample(examples, batch_size)


def evaluate_model(
    ppo_trainer: Any,
    tokenizer: Any,
    val_examples: list[Example],
    args: argparse.Namespace,
) -> dict[str, float]:
    if not val_examples:
        return {"acc": 0.0, "macro_f1": 0.0, "parse_rate": 0.0}

    subset = val_examples[: args.eval_max_examples]
    counts = Counter()
    conf = Counter()
    for ex in tqdm(subset, desc="Val eval", leave=False):
        prompt = build_prompt(tokenizer, ex.prompt)
        toks = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_tokens,
        ).to(ppo_trainer.accelerator.device)
        with torch.no_grad():
            out = ppo_trainer.model.generate(
                **toks,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][toks["input_ids"].shape[1] :]
        txt = tokenizer.decode(gen, skip_special_tokens=True)
        parsed = parse_prediction(txt)
        if parsed is None:
            pred = "PARSE_FAIL"
        else:
            pred = str(parsed.get("evaluation", "")).upper()
            if pred not in LABELS:
                pred = "PARSE_FAIL"
        counts["n"] += 1
        counts["parse_ok"] += float(pred != "PARSE_FAIL")
        if pred in LABELS and pred == ex.evaluation:
            counts["correct"] += 1
        conf[(ex.evaluation, pred)] += 1

    def f1_for_label(lbl: str) -> float:
        tp = conf[(lbl, lbl)]
        fp = sum(conf[(g, lbl)] for g in LABELS if g != lbl)
        fn = sum(conf[(lbl, p)] for p in LABELS if p != lbl)
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        if tp == 0 and fp == 0 and fn == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    macro_f1 = sum(f1_for_label(lbl) for lbl in LABELS) / len(LABELS)
    return {
        "acc": counts["correct"] / max(1, counts["n"]),
        "macro_f1": macro_f1,
        "parse_rate": counts["parse_ok"] / max(1, counts["n"]),
    }


def train(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Preparing PPO dataset ...")
    if args.cache_prompts_path:
        cache_path = Path(args.cache_prompts_path)
        cached = [Example(**x) for x in json.loads(cache_path.read_text(encoding="utf-8"))]
        examples = cached
        print(f"Loaded {len(examples):,} prompt records from cache")
    else:
        examples = convert_training_set_to_examples(
            training_set_path=Path(args.training_set_path),
            paper_root=Path(args.paper_root),
            max_context_chars=args.max_context_chars,
        )
    if not examples:
        raise RuntimeError("No PPO examples built. Check --paper_root and input paths.")

    train_examples, val_examples = split_by_doi(examples, args.val_ratio, args.seed)
    if not train_examples:
        raise RuntimeError("Empty train split after DOI split.")
    print(f"Train examples: {len(train_examples):,}")
    print(f"Val examples:   {len(val_examples):,}")

    weights = class_weights(train_examples, args.class_balance_alpha)
    print(f"Class weights: {weights}")

    print("Loading 4-bit base model ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )

    print(f"Loading SFT adapter from {args.sft_adapter_path} ...")
    peft_model = PeftModel.from_pretrained(base_model, args.sft_adapter_path, is_trainable=True)
    peft_model.config.use_cache = False

    print("Wrapping model with value head for PPO ...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
    ref_model = create_reference_model(model)

    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        target_kl=args.target_kl,
        init_kl_coef=args.init_kl_coef,
        seed=args.seed,
        log_with="wandb",
        project_kwargs={"project": "cs285-ppo-evaluator"},
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,
        data_collator=None,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    rnd = random.Random(args.seed)
    print("Starting PPO updates ...")
    for step in range(1, args.total_ppo_updates + 1):
        batch_examples = sample_batch(train_examples, args.batch_size, rnd)

        prompts = [build_prompt(tokenizer, ex.prompt) for ex in batch_examples]
        query_tensors = []
        for ptxt in prompts:
            ids = tokenizer(
                ptxt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            )["input_ids"][0]
            query_tensors.append(ids.to(ppo_trainer.accelerator.device))

        response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
        decoded = [tokenizer.decode(rt, skip_special_tokens=True) for rt in response_tensors]

        rewards = []
        reward_breakdown = Counter()
        for ex, txt in zip(batch_examples, decoded):
            parsed = parse_prediction(txt)
            reward_value, metrics = reward_for_prediction(parsed, ex, weights, args)
            rewards.append(torch.tensor(reward_value, dtype=torch.float32).to(ppo_trainer.accelerator.device))
            reward_breakdown["reward_sum"] += reward_value
            reward_breakdown["format"] += metrics["format"]
            reward_breakdown["consistency"] += metrics["consistency"]
            reward_breakdown["label"] += metrics["label"]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % args.log_every == 0:
            bsz = max(1, len(batch_examples))
            avg_reward = reward_breakdown["reward_sum"] / bsz
            avg_format = reward_breakdown["format"] / bsz
            avg_consistency = reward_breakdown["consistency"] / bsz
            avg_label = reward_breakdown["label"] / bsz
            kl_val = float(stats.get("objective/kl", math.nan))
            print(
                f"[step {step:5d}] reward={avg_reward:+.4f} "
                f"format={avg_format:.3f} consistency={avg_consistency:.3f} "
                f"label={avg_label:+.3f} kl={kl_val:.4f}"
            )

        if step % args.eval_every == 0 and val_examples:
            metrics = evaluate_model(ppo_trainer, tokenizer, val_examples, args)
            print(
                f"[val @ {step:5d}] "
                f"acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                f"parse_rate={metrics['parse_rate']:.4f}"
            )

    final_dir = out_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving PPO adapter to {final_dir} ...")
    model.pretrained_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
