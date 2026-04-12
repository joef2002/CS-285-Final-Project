# CS 285 Final Project

Supervised fine-tuning of **Qwen3.5-4B** (4-bit LoRA) on chemistry question-answer pairs derived from scientific literature.

## Repository structure

```
sft_train.py          # SFT training script (LoRA, 4-bit, HF Trainer)
ppo_train.py          # PPO training from SFT adapter (class-balanced reward)
ppo_eval.py           # Parse-aware evaluator metrics for TP/TN/FP/FN
modal_train.py        # Modal cloud GPU launcher (H100 × 4, DDP)
pyproject.toml        # Dependencies and project metadata
training_set.json     # Source training data (241 papers, 8,986 QA pairs)
training_qwen.jsonl   # Chat-formatted JSONL for Qwen SFT (7,552 samples)
test_set.json         # Human-verified evaluation set (57 papers, 547 QA pairs)
single_hop_dataset/   # Single-hop QA evaluation JSONs
final_adapter/        # Fine-tuned LoRA adapter (config only in repo)
```

## Training details

- **Model:** Qwen/Qwen3.5-4B, 4-bit NF4 quantization (bitsandbytes)
- **Method:** LoRA (r=16, α=32, dropout=0.05)
- **Targets:** q/k/v/o/gate/up/down projections
- **Trainable params:** 21.2M / 4.2B total (0.5%)
- **Hardware:** 4× H100 80GB (DDP via torchrun)
- **Hyperparams:** batch=1/GPU, grad_accum=2, lr=5e-5, cosine schedule, 1 epoch
- **Max sequence length:** 16,384 tokens
- **Loss:** cross-entropy on assistant tokens only (prefix masked with -100)
- **Final train loss:** 0.452 | **Eval loss:** 0.353


## Load for inference

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("final_adapter/final_adapter", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("final_adapter/final_adapter")
```

## PPO from SFT (TP/TN/FP/FN evaluator)

This repo now includes a PPO entrypoint (`ppo_train.py`) that initializes from your SFT LoRA adapter and optimizes a class-balanced reward on final label correctness.

### PPO reward design

- **Format reward:** parsed output is valid JSON with `grounded`, `correct`, and `evaluation`.
- **Consistency reward:** `(grounded, correct)` matches `evaluation` (`TP`/`TN`/`FP`/`FN` mapping).
- **Label reward:** predicted `evaluation` matches the gold label; weighted by inverse-frequency class weights to upweight rare classes.

### Run PPO training

Install dependencies (including `trl`) and run:

```bash
uv sync --extra remote
python ppo_train.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --sft_adapter_path final_adapter/final_adapter \
  --training_qwen_path training_qwen.jsonl \
  --output_dir runs/ppo_from_sft \
  --total_ppo_updates 1200 \
  --batch_size 8 \
  --mini_batch_size 2 \
  --learning_rate 1e-6 \
  --eval_every 100
```

Notes:

- Preferred path: use `--training_qwen_path training_qwen.jsonl` (context already embedded, no MS/SI file lookup needed).
- If you use `--training_set_path` instead, `--paper_root` must point to the directory that contains files referenced by `ms` and `si`.
- `--sft_adapter_path` must point to a real adapter checkpoint containing `adapter_model.safetensors` (or `.bin`), not config-only files.
- Data split is by DOI (paper-level) to reduce leakage.

### Evaluate PPO checkpoint

```bash
python ppo_eval.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --adapter_path runs/ppo_from_sft/final_adapter \
  --test_path test_qwen.jsonl \
  --output_path runs/ppo_from_sft/test_predictions.jsonl
```

This reports:

- accuracy
- macro-F1
- parse rate
- per-class precision/recall for TP/TN/FP/FN

### Run PPO on Modal (`python3`)

`modal_train.py` includes `ppo_remote` and `ppo_eval_remote` for GPU runs.

Recommended: run PPO from `training_qwen.jsonl` (no raw corpus mount needed).
Legacy option: `training_set.json` stores `ms`/`si` as relative paths like `all paper ft data/...`; since `.gitignore` excludes that folder, copy corpus to Modal volume and set `--paper_root /vol`.

Before training, verify path mapping with:

```bash
uv run modal run modal_train.py::ppo_data_check_remote -- \
  --training_qwen_path /root/project/training_qwen.jsonl \
  --sample_papers 50
```

```bash
# 1) Train PPO on Modal (uses python3 inside container)
uv run modal run modal_train.py::ppo_remote -- \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --sft_adapter_path /root/project/final_adapter/final_adapter \
  --training_qwen_path /root/project/training_qwen.jsonl \
  --output_dir /vol/runs/ppo_from_sft \
  --total_ppo_updates 1200 \
  --batch_size 8 \
  --mini_batch_size 2 \
  --learning_rate 1e-6 \
  --eval_every 100

# 2) Evaluate PPO adapter on test_qwen
uv run modal run modal_train.py::ppo_eval_remote -- \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --adapter_path /vol/runs/ppo_from_sft/final_adapter \
  --test_path /root/project/test_qwen.jsonl \
  --output_path /vol/runs/ppo_from_sft/test_predictions.jsonl
```

If you use `--training_set_path` mode and `--paper_root` is wrong, PPO logs show many skipped papers due to unreadable MS/SI context.

---

## GRPO from SFT (Group Relative Policy Optimization)

This repo also includes a GRPO entrypoint (`grpo_train.py`) that initializes from your SFT LoRA adapter and optimizes a grouped objective over multiple sampled completions per prompt.

### GRPO objective and reward design

- **Grouped sampling:** for each prompt, sample `group_size` completions.
- **Relative advantages:** rewards are normalized within each prompt group (mean/std), then used as per-sample advantages.
- **Clipped policy update:** GRPO uses a PPO-style clipped objective (`--grpo_clip_range`) over sequence log-probs.
- **Adaptive KL penalty:** KL coefficient is adjusted toward `--target_kl` from `--init_kl_coef`.
- **Reward components:** same parse-aware/class-balanced signal family as PPO:
  - format reward (valid JSON schema)
  - consistency reward (`grounded`/`correct` agrees with `evaluation`)
  - label reward (class-balanced TP/TN/FP/FN correctness)

### Run GRPO training

Install dependencies and run:

```bash
uv sync --extra remote
python grpo_train.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --sft_adapter_path final_adapter/final_adapter \
  --training_qwen_path training_qwen.jsonl \
  --output_dir runs/grpo_from_sft \
  --total_grpo_updates 1200 \
  --batch_size 1 \
  --group_size 4 \
  --mini_batch_size 1 \
  --grpo_epochs 1 \
  --logprob_batch_size 1 \
  --max_prompt_tokens 512 \
  --max_new_tokens 128 \
  --max_context_chars 1500 \
  --learning_rate 1e-6 \
  --eval_every 100
```

Notes:

- Preferred path: use `--training_qwen_path training_qwen.jsonl` (context already embedded, no MS/SI file lookup needed).
- If you use `--training_set_path` instead, `--paper_root` must point to the directory that contains files referenced by `ms` and `si`.
- `--sft_adapter_path` must point to a real adapter checkpoint containing `adapter_model.safetensors` (or `.bin`), not config-only files.
- GRPO defaults to a single visible GPU (`CUDA_VISIBLE_DEVICES=0`) for rollout/update stability and lower memory pressure.

### Evaluate GRPO checkpoint

```bash
python grpo_eval.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --adapter_path runs/grpo_from_sft/final_adapter \
  --test_path test_qwen.jsonl \
  --output_path runs/grpo_from_sft/test_predictions.jsonl
```

This reports:

- accuracy
- macro-F1
- parse rate
- per-class precision/recall for TP/TN/FP/FN

### Run GRPO on Modal (`python3`)

`modal_train.py` includes `grpo_remote` and `grpo_eval_remote` for GPU runs.

Recommended: run GRPO from `training_qwen.jsonl` (no raw corpus mount needed).
Legacy option: `training_set.json` stores `ms`/`si` as relative paths like `all paper ft data/...`; since `.gitignore` excludes that folder, copy corpus to Modal volume and set `--paper_root /vol`.

Before training, verify path mapping with:

```bash
uv run modal run modal_train.py::grpo_data_check_remote -- \
  --training_qwen_path /root/project/training_qwen.jsonl \
  --sample_papers 50
```

```bash
# 1) Train GRPO on Modal (uses python3 inside container)
uv run modal run modal_train.py::grpo_remote -- \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --sft_adapter_path /root/project/final_adapter/final_adapter \
  --training_qwen_path /root/project/training_qwen.jsonl \
  --output_dir /vol/runs/grpo_from_sft \
  --total_grpo_updates 1200 \
  --batch_size 1 \
  --group_size 4 \
  --mini_batch_size 1 \
  --grpo_epochs 1 \
  --logprob_batch_size 1 \
  --max_prompt_tokens 512 \
  --max_new_tokens 128 \
  --max_context_chars 1500 \
  --learning_rate 1e-6 \
  --eval_every 100

# 2) Evaluate GRPO adapter on test_qwen
uv run modal run modal_train.py::grpo_eval_remote -- \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --adapter_path /vol/runs/grpo_from_sft/final_adapter \
  --test_path /root/project/test_qwen.jsonl \
  --output_path /vol/runs/grpo_from_sft/test_predictions.jsonl
```

If you use `--training_set_path` mode and `--paper_root` is wrong, GRPO logs show many skipped papers due to unreadable MS/SI context.

---

## Dataset structure and statistics

### `training_qwen.jsonl`

JSONL file derived from `training_set.json`, formatted for Qwen chat-template SFT. Each line is a JSON object with a `"messages"` array of `{role, content}` dicts.

#### Source schema (`training_set.json`)

```json
{
  "doi": "...",
  "ms": "...",
  "si": "...",
  "QA pairs": [
    {
      "question": "...",
      "answer": "...",
      "evaluation": "...",
      "grounded": true,
      "correct": true,
      "explanation": "...",
      "data_source": "...",
      "human_verified": false
    }
  ]
}
```

| Field | Description |
|--------|-------------|
| **`question`** / **`answer`** | From `combined_summary_fixed.xlsx`. |
| **`evaluation`** | From the workbook **`evaluation`** column (same four values as test **`label`**: `TP`, `TN`, `FP`, `FN`). |
| **`grounded`** / **`correct`** | Derived from **`evaluation`** using the same 2×2 table as the test set (above). |
| **`explanation`** | From the **`explanation_*`** column of the first model (in priority order) whose **`evaluation_*`** matches **`evaluation`**: Claude → GPT o1/o3 → GPT → Gemini. |
| **`data_source`** | From **`data_source`** column. |
| **`human_verified`** | Always **`false`**. |

Training rows that match **`test_set.json`** by `(doi, normalized question)` are **excluded** when building `training_set.json` (see `build_training_set.py`).

#### Filtering

Samples exceeding an estimated 16,384 tokens (~65,536 characters) were removed to fit within a single H100 80 GB GPU during LoRA fine-tuning.

#### Statistics

| Metric | Value |
|--------|------:|
| Papers (array length) | 241 |
| QA pairs (total) | 8,986 |
| Papers with zero QA pairs after filtering | 0 |

**`evaluation` distribution**

| Evaluation | Count | Share |
|------------|------:|------:|
| TP | 4,319 | 48.1% |
| FP | 2,633 | 29.3% |
| TN | 1,626 | 18.1% |
| FN | 408 | 4.5% |

### `test_set.json`

Human-verified evaluation set. JSON array of objects.

```json
{
  "doi": "...",
  "ms": "...",
  "si": "...",
  "QA pairs": [
    {
      "question": "...",
      "answer": "...",
      "label": "...",
      "grounded": true,
      "correct": true,
      "human_verified": true
    }
  ]
}
```

| Field | Description |
|--------|-------------|
| **`doi`** | Canonical DOI for the publication. |
| **`ms`** | Path to main manuscript combined text, or `""`. |
| **`si`** | Path(s) to supplementary combined text (newline-separated), or `""`. |
| **`label`** | Human judgment: `TP`, `TN`, `FP`, or `FN`. |
| **`grounded`** | Whether the pair is grounded in the source text. |
| **`correct`** | Whether the answer is correct relative to that grounding. |
| **`human_verified`** | Always `true`. |

**`grounded` / `correct` from `label`**

| Grounded | Correct | `label` |
|:--------:|:-------:|---------|
| Yes | Yes | `TP` |
| Yes | No | `FP` |
| No | Yes | `TN` |
| No | No | `FN` |

#### Statistics

| Metric | Value |
|--------|------:|
| Papers | 57 |
| QA pairs | 547 |

**`label` distribution**

| Label | Count | Share |
|-------|------:|------:|
| TP | 487 | 89.0% |
| TN | 37 | 6.8% |
| FP | 21 | 3.8% |
| FN | 2 | 0.4% |

The test set is heavily skewed toward **TP**; negative and error classes are much rarer.
