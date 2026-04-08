import argparse
import json
import os
import torch

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SFT on QA data, continuing from DAPT adapter"
    )

    # Model & data
    p.add_argument("--model_name_or_path", type=str,
                   default="Qwen/Qwen3.5-4B")
    p.add_argument("--data_path", type=str,
                   default="/workspace/data/finetune_train.jsonl")
    p.add_argument("--output_dir", type=str,
                   default="/workspace/output/sft_dapt_no_refs")

    # Sequence
    p.add_argument("--max_seq_len", type=int, default=16384)

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)

    # Logging & saving
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)

    # Data split
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def format_messages_to_chat(messages):
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)

def load_qa_dataset(path):
    from datasets import Dataset

    texts = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(format_messages_to_chat(obj["messages"]))

    return Dataset.from_dict({"text": texts})

# ---------------------------------------------------------------------------
# Tokenization with masked loss
# ---------------------------------------------------------------------------

def tokenize_with_masked_loss(examples, tokenizer, max_seq_len):
    all_input_ids = []
    all_labels = []

    assistant_marker = "<|im_start|>assistant\n"

    for text in examples["text"]:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        assistant_start_pos = text.find(assistant_marker)
        if assistant_start_pos == -1:
            # Fallback: compute loss on everything
            all_input_ids.append(input_ids)
            all_labels.append(input_ids.copy())
            continue
        prefix_text = text[:assistant_start_pos + len(assistant_marker)]
        prefix_ids = tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )["input_ids"]
        labels = [-100] * len(input_ids)
        for j in range(len(prefix_ids), len(input_ids)):
            labels[j] = input_ids[j]
        all_input_ids.append(input_ids)
        all_labels.append(labels)
    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
    }

# ---------------------------------------------------------------------------
# Dynamic padding collator
# ---------------------------------------------------------------------------

class DynamicPaddingCollatorSFT:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(args):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import prepare_model_for_kbit_training
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Loading QA dataset from {args.data_path} ...")
    ds = load_qa_dataset(args.data_path)
    print(f"  Total samples: {len(ds):,}")
    split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    print("Tokenizing ...")
    tok_kwargs = dict(
        batched=True,
        remove_columns=["text"],
        num_proc=min(os.cpu_count() or 1, 8),
        desc="Tokenizing",
    )
    train_ds = train_ds.map(
        lambda ex: tokenize_with_masked_loss(ex, tokenizer, args.max_seq_len),
        **tok_kwargs,
    )
    val_ds = val_ds.map(
        lambda ex: tokenize_with_masked_loss(ex, tokenizer, args.max_seq_len),
        **tok_kwargs,
    )
    total_answer_tokens = sum(
        sum(1 for l in lab if l != -100) for lab in train_ds["labels"]
    )
    print(f"  Answer tokens (loss computed on): {total_answer_tokens:,}")
    print("Loading base model with 4-bit quantization ...")
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
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=not args.no_gradient_checkpointing
    )
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    gc_kwargs = {}
    if use_gc:
        gc_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=use_gc,
        bf16=args.bf16,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"sft-{args.model_name_or_path.split('/')[-1]}",
        seed=args.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        **gc_kwargs,
    )
    collator = DynamicPaddingCollatorSFT(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    final_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving adapter to {final_dir} ...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
