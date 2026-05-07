#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter_path", type=str,
                   default="runs/dpo_from_sft/final_adapter")
    p.add_argument("--test_path", type=str, default="test_qwen.jsonl")
    p.add_argument("--output_path", type=str,
                   default="runs/dpo_from_sft/test_predictions.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_input_tokens", type=int, default=16384)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )

    print(f"Loading DPO adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    print(f"Loading test data from {args.test_path} ...")
    test_data = []
    with open(args.test_path) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} examples")

    with open(args.output_path, "w") as out:
        for i, example in enumerate(tqdm(test_data, desc="DPO inference")):
            messages = example["messages"]
            expected = example.get("expected", {})

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=args.max_input_tokens,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            out.write(json.dumps({
                "index": i,
                "expected": expected,
                "model_response": response_text,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(test_data)} results to {args.output_path}")


if __name__ == "__main__":
    main()
