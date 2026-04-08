# CS 285 Final Project

Supervised fine-tuning of **Qwen3.5-4B** (4-bit LoRA) on chemistry question-answer pairs derived from scientific literature.

## Repository structure

```
sft_train.py          # SFT training script (LoRA, 4-bit, HF Trainer)
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
