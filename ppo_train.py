#!/usr/bin/env python3
"""PPO fine-tuning for the reticular-chemistry evaluator from an SFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep PPO execution on a single visible GPU by default.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    p.add_argument("--training_qwen_path", type=str, default="")
    p.add_argument("--paper_root", type=str, default=".")
    p.add_argument("--output_dir", type=str, default="runs/ppo_from_sft")
    p.add_argument("--cache_prompts_path", type=str, default="")

    # Data
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_context_chars", type=int, default=4000)
    p.add_argument("--max_prompt_tokens", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--bf16", action="store_true", default=True)

    # PPO
    p.add_argument("--total_ppo_updates", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mini_batch_size", type=int, default=2)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--ppo_clip_range", type=float, default=0.2)
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


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.S)
    return m.group(1).strip() if m else ""


def _parse_yes_no_bool(text: str, key: str) -> bool | None:
    m = re.search(rf"{key}\s*:\s*(yes|no|true|false)", text, flags=re.I)
    if not m:
        return None
    v = m.group(1).lower()
    return v in {"yes", "true"}


def _parse_label(text: str) -> str | None:
    m = re.search(r"\bEvaluation\s*:\s*(TP|TN|FP|FN)\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    # Fallback for legacy outputs that may only contain the label.
    m = re.search(r"\b(TP|TN|FP|FN)\b", text)
    return m.group(1).upper() if m else None


def convert_training_qwen_to_examples(training_qwen_path: Path, max_context_chars: int) -> list[Example]:
    if not training_qwen_path.is_file():
        # Modal-specific fallback: users may pass /vol/training_qwen.jsonl,
        # but this file usually comes from the project mount at /root/project.
        alt = Path("/root/project") / training_qwen_path.name
        if alt.is_file():
            print(
                f"[warning] training_qwen path not found at {training_qwen_path}; "
                f"falling back to {alt}"
            )
            training_qwen_path = alt

    examples: list[Example] = []
    skipped = 0
    with training_qwen_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Building PPO examples from training_qwen")):
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 3:
                skipped += 1
                continue

            user_msg = None
            assistant_msg = None
            for m in messages:
                if m.get("role") == "user":
                    user_msg = str(m.get("content", ""))
                elif m.get("role") == "assistant":
                    assistant_msg = str(m.get("content", ""))
            if not user_msg or not assistant_msg:
                skipped += 1
                continue

            # training_qwen rows contain a full user prompt with <context>/<question>/<answer>.
            # Rebuild prompt with a bounded context segment to keep PPO generation tensors tractable.
            context = _extract_tag(user_msg, "context")
            question = _extract_tag(user_msg, "question")
            answer = _extract_tag(user_msg, "answer")
            if context and question and answer:
                user_msg = build_user_prompt(context[:max_context_chars], question, answer)
            elif len(user_msg) > (max_context_chars + 4096):
                # Conservative fallback if row is malformed: keep tail where question/answer usually live.
                user_msg = user_msg[-(max_context_chars + 4096) :]

            label = _parse_label(assistant_msg)
            if label not in LABELS:
                skipped += 1
                continue

            grounded = _parse_yes_no_bool(assistant_msg, "Grounded")
            correct = _parse_yes_no_bool(assistant_msg, "Correct")
            if grounded is None or correct is None:
                # Fallback to label mapping if booleans are absent.
                for (g, c), l in PAIR_TO_LABEL.items():
                    if l == label:
                        grounded, correct = g, c
                        break
            if grounded is None or correct is None:
                skipped += 1
                continue

            doi = str(obj.get("doi", "")).strip()
            if not doi:
                context = _extract_tag(user_msg, "context")
                if context:
                    doi = "ctx_" + hashlib.md5(context.encode("utf-8")).hexdigest()[:16]
                else:
                    doi = f"row_{idx}"

            examples.append(
                Example(
                    doi=doi,
                    prompt=user_msg,
                    grounded=grounded,
                    correct=correct,
                    evaluation=label,
                )
            )

    print(f"Built {len(examples):,} PPO examples from training_qwen.jsonl")
    print(f"Skipped malformed rows: {skipped:,}")
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
    blob = extract_first_json_object(_strip_think_prefix(response_text))
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
    model: Any,
    tokenizer: Any,
    device: torch.device,
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
        ).to(device)
        with torch.no_grad():
            out = model.generate(
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


def _strip_think_prefix(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _sequence_log_probs(
    model: Any,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    if not sequences:
        return torch.empty(0, device=device)

    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    completion_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.float32, device=device)

    for i, (seq, p_len) in enumerate(zip(sequences, prompt_lengths, strict=True)):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :seq_len] = 1
        start = max(p_len - 1, 0)
        end = max(seq_len - 1, 0)
        if end > start:
            completion_mask[i, start:end] = 1.0

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    token_log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(token_log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (gathered * completion_mask).sum(dim=1)


def _validate_sft_adapter_path_or_raise(path_value: str) -> None:
    p = Path(path_value)
    if not p.exists():
        # Could still be a Hub repo id, so only error hard for absolute/local-looking paths.
        if path_value.startswith("/") or path_value.startswith("."):
            raise RuntimeError(
                f"SFT adapter path does not exist: {path_value}. "
                "Provide a valid local path containing adapter weights or a valid HF repo id."
            )
        return

    if not p.is_dir():
        raise RuntimeError(f"SFT adapter path is not a directory: {path_value}")

    has_cfg = (p / "adapter_config.json").is_file()
    has_weights = (p / "adapter_model.safetensors").is_file() or (p / "adapter_model.bin").is_file()
    if has_cfg and not has_weights:
        raise RuntimeError(
            "SFT adapter directory is missing adapter weights. Found adapter_config.json but no "
            "adapter_model.safetensors/adapter_model.bin at "
            f"{path_value}. "
            "Use a real SFT checkpoint directory (for example from /vol/runs/.../final_adapter) as --sft_adapter_path."
        )


def train(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
        if torch.cuda.device_count() > 1:
            # Prevent transformers Trainer from wrapping model in DataParallel.
            print("[warning] More than one CUDA device visible; forcing training to cuda:0.")
            torch.cuda.set_device(0)

    # Guardrail: very long prompts with larger PPO batches can trigger large
    # conv1d tensors in Qwen's linear attention path.
    effective_max_context_chars = args.max_context_chars
    if args.batch_size >= 8 and args.max_context_chars > 4000:
        effective_max_context_chars = 4000
        print(
            "[warning] Reducing max_context_chars to 4000 for stability with "
            f"batch_size={args.batch_size}. You can override by lowering batch size."
        )

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
    elif args.training_qwen_path:
        examples = convert_training_qwen_to_examples(
            training_qwen_path=Path(args.training_qwen_path),
            max_context_chars=effective_max_context_chars,
        )
    else:
        examples = convert_training_set_to_examples(
            training_set_path=Path(args.training_set_path),
            paper_root=Path(args.paper_root),
            max_context_chars=effective_max_context_chars,
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
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    print(f"Loading SFT adapter from {args.sft_adapter_path} ...")
    _validate_sft_adapter_path_or_raise(args.sft_adapter_path)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter_path, is_trainable=True)
    model.config.use_cache = False
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in adapter model.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    kl_coef = float(args.init_kl_coef)
    rnd = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)

    print("Starting PPO optimization loop ...")
    for update in range(1, args.total_ppo_updates + 1):
        batch_examples = sample_batch(train_examples, args.batch_size, rnd)
        prompts = [build_prompt(tokenizer, ex.prompt) for ex in batch_examples]
        prompt_toks = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        ).to(device)
        prompt_input_ids = prompt_toks["input_ids"]
        prompt_attention_mask = prompt_toks["attention_mask"]
        padded_prompt_len = prompt_input_ids.shape[1]

        model.eval()
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = generated[:, padded_prompt_len:]

        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        rewards: list[float] = []
        for ex, p_ids, p_mask, c_ids in zip(
            batch_examples, prompt_input_ids, prompt_attention_mask, completion_ids, strict=True
        ):
            prompt_list = p_ids[p_mask.bool()].tolist()
            completion_list = c_ids.tolist()
            if tokenizer.pad_token_id is not None:
                while completion_list and completion_list[-1] == tokenizer.pad_token_id:
                    completion_list.pop()
            if not completion_list and tokenizer.eos_token_id is not None:
                completion_list = [tokenizer.eos_token_id]

            completion_text = _strip_think_prefix(
                tokenizer.decode(completion_list, skip_special_tokens=True)
            )
            parsed = parse_prediction(completion_text)
            reward_value, _metrics = reward_for_prediction(parsed, ex, weights, args)

            sequences.append(prompt_list + completion_list)
            prompt_lengths.append(len(prompt_list))
            rewards.append(float(reward_value))

        with torch.no_grad():
            old_log_probs = _sequence_log_probs(
                model=model,
                sequences=sequences,
                prompt_lengths=prompt_lengths,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = rewards_t - rewards_t.mean()
        advantages = advantages / (advantages.std(unbiased=False) + 1e-6)

        minibatch_indices = list(range(len(sequences)))
        model.train()
        step_approx_kls: list[float] = []
        accum = 0
        for _epoch in range(args.ppo_epochs):
            rnd.shuffle(minibatch_indices)
            for start in range(0, len(minibatch_indices), args.mini_batch_size):
                idx = minibatch_indices[start : start + args.mini_batch_size]
                mb_sequences = [sequences[i] for i in idx]
                mb_prompt_lengths = [prompt_lengths[i] for i in idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_adv = advantages[idx]

                new_log_probs = _sequence_log_probs(
                    model=model,
                    sequences=mb_sequences,
                    prompt_lengths=mb_prompt_lengths,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - args.ppo_clip_range, 1.0 + args.ppo_clip_range
                )
                policy_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()
                approx_kl = 0.5 * ((new_log_probs - mb_old_log_probs) ** 2).mean()
                loss = policy_loss + kl_coef * approx_kl
                loss.backward()

                accum += 1
                if accum % max(1, args.gradient_accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step_approx_kls.append(float(approx_kl.detach().cpu().item()))

        if accum % max(1, args.gradient_accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        mean_reward = float(rewards_t.mean().detach().cpu().item())
        mean_kl = float(sum(step_approx_kls) / max(1, len(step_approx_kls)))
        if mean_kl > args.target_kl * 1.5:
            kl_coef = min(1.0, kl_coef * 1.5)
        elif mean_kl < args.target_kl / 1.5:
            kl_coef = max(1e-4, kl_coef / 1.5)

        if update % args.log_every == 0 or update == 1:
            print(
                f"[update {update:05d}] reward={mean_reward:.4f} "
                f"approx_kl={mean_kl:.4f} kl_coef={kl_coef:.6f}"
            )

        if val_examples and (update % args.eval_every == 0):
            model.eval()
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                device=device,
                val_examples=val_examples,
                args=args,
            )
            print(
                f"[eval {update:05d}] "
                f"acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                f"parse_rate={metrics['parse_rate']:.4f}"
            )
            ckpt_dir = out_dir / f"checkpoint-{update}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    final_dir = out_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving PPO adapter to {final_dir} ...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
