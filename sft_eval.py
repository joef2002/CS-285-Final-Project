#!/usr/bin/env python3
"""Run SFT model inference on test_qwen.jsonl and write predictions to a JSONL file."""

import argparse
import json
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", type=str,
                   default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter_path", type=str,
                   default="final_adapter/final_adapter")
    p.add_argument("--test_path", type=str,
                   default="test_qwen.jsonl")
    p.add_argument("--output_path", type=str,
                   default="test_results.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=512)
    return p.parse_args()


def main():
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

    print(f"Loading adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Load test data
    print(f"Loading test data from {args.test_path} ...")
    test_data = []
    with open(args.test_path) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"  {len(test_data)} examples")

    # Run inference and write results
    with open(args.output_path, "w") as out:
        for i, example in enumerate(tqdm(test_data, desc="Inference")):
            messages = example["messages"]
            expected = example.get("expected", {})

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
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

            record = {
                "index": i,
                "expected": expected,
                "model_response": response_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(test_data)} results to {args.output_path}")


if __name__ == "__main__":
    main()
