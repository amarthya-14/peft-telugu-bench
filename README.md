# peft-telugu-bench

How much do LoRA, DoRA and QLoRA differ when fine-tuning a small LLM for Telugu, in quality and in cost?

## Setup

| | |
|---|---|
| Model | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), chat template, thinking disabled |
| Data | Telugu subset of [`CohereLabs/aya_dataset`](https://huggingface.co/datasets/CohereLabs/aya_dataset) (human-written, Apache-2.0): 8,439 train / 250 test |
| Splits | 300 examples of Aya train held out for validation; Aya test used only for the final number |
| Adapters | rank 16, alpha 32, dropout 0.05, all linear layers |
| Training | 1 epoch, lr 2e-4 cosine, effective batch 16 (2 × 8 accumulation), loss on answer tokens only |

### Telugu is expensive to tokenize

The Qwen3 tokenizer gets ~0.7 characters per token on this data (English is typically 3–4), so the median example is 772 tokens. Examples over 1024 tokens are dropped rather than truncated (61% of the data is kept) so no answer is cut mid-sentence.

## Results

| Method | Trainable params | Peak GPU mem | Train time | Test loss | Test ppl | Adapter size |
|---|---|---|---|---|---|---|
| LoRA | 10.1M (1.67%) | 5.24 GB | 115 min | 0.760 | 2.14 | 40.4 MB |
| DoRA | | | | | | |
| QLoRA | | | | | | |

_GPU: Tesla T4 (Colab), 4,982 training examples. PyTorch reports bf16 as supported on the T4 (emulated), so runs use bf16 weights and bf16 autocast; all three methods use the same setting. Each run writes `results/<method>-r16/metrics.json`._

LoRA validation loss: 0.862 (step 100), 0.821 (200), 0.811 (300), 0.811 (end), so it has levelled off by the end of the single epoch. Test loss is lower than validation loss because the two sets come from different Aya splits, so the two numbers shouldn't be compared directly.

## Run

```bash
uv sync
uv run python train.py --method lora     # or dora, qlora (qlora needs an NVIDIA GPU)
```

Smoke test on a laptop (tiny random model, a few steps):

```bash
uv run python train.py --model trl-internal-testing/tiny-Qwen3ForCausalLM \
  --max_train_samples 32 --max_steps 3 --batch_size 2 --grad_accum 1
```

On Colab, use `colab_train.ipynb`; on Kaggle, `kaggle_train.ipynb` (runs DoRA and QLoRA in the background).

## Next

- Generation-based eval (chrF on test answers), not just loss
- Rank sweep (8 / 16 / 64)
