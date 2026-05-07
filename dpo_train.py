#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--sft_adapter_path", type=str,
                   default="final_adapter/final_adapter")
    p.add_argument("--dpo_pairs_path", type=str, default="dpo_pairs.jsonl")
    p.add_argument("--output_dir", type=str, default="runs/dpo_from_sft")

    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss_type", type=str, default="sigmoid",
                   choices=["sigmoid", "ipo", "hinge"])
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--max_prompt_length", type=int, default=15872)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--precompute_ref_log_probs", type=lambda v: str(v).lower() in {"1", "true", "yes"},
                   default=True)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    return p.parse_args()


def load_dpo_dataset(path: str):
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            obj.pop("_meta", None)
            rows.append(obj)
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig
    from trl import DPOTrainer, DPOConfig

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model with 4-bit quantization ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    chosen_attn = args.attn_implementation
    if chosen_attn == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            print(f"[dpo_train] flash_attention_2 unavailable ({e!r}); falling back to sdpa.")
            chosen_attn = "sdpa"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=chosen_attn,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    print(f"[dpo_train] attention backend = {chosen_attn}")

    print(f"Loading SFT adapter from {args.sft_adapter_path} as initial policy ...")
    model = PeftModel.from_pretrained(
        base, args.sft_adapter_path, is_trainable=True,
    )
    model.print_trainable_parameters()

    ref_model = None

    print(f"Loading DPO pairs from {args.dpo_pairs_path} ...")
    ds = load_dpo_dataset(args.dpo_pairs_path)
    print(f"  Total pairs: {len(ds):,}")
    split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds):,}  |  Val: {len(eval_ds):,}")

    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    gc_kwargs = {}
    if use_gc:
        gc_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    run_name = args.run_name or f"dpo-{Path(args.model_name_or_path).name}-beta{args.beta}"

    dpo_kwargs = dict(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=use_gc,
        bf16=args.bf16,
        optim=args.optim,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        run_name=run_name,
        seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        **gc_kwargs,
    )

    import inspect
    sig_params = inspect.signature(DPOConfig).parameters
    if "max_prompt_length" in sig_params:
        dpo_kwargs["max_prompt_length"] = args.max_prompt_length
    else:
        print(
            f"[dpo_train] DPOConfig in this TRL version does not accept "
            f"max_prompt_length; relying on max_length={args.max_length} for truncation."
        )
    if "precompute_ref_log_probs" in sig_params:
        dpo_kwargs["precompute_ref_log_probs"] = args.precompute_ref_log_probs
        if args.precompute_ref_log_probs:
            print(
                "[dpo_train] precompute_ref_log_probs=True: caching reference "
                "log-probs in a one-time pre-pass to save training-step memory."
            )
    else:
        print("[dpo_train] DPOConfig in this TRL version has no precompute_ref_log_probs option.")

    dpo_config = DPOConfig(**dpo_kwargs)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting DPO training ...")
    trainer.train()

    final_dir = Path(args.output_dir) / "final_adapter"
    print(f"Saving DPO adapter to {final_dir} ...")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
