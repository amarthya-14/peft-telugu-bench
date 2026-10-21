"""Fine-tune a small LLM on Telugu instructions with LoRA, DoRA or QLoRA and record cost/quality metrics.

    python train.py --method lora
    python train.py --method qlora      # needs an NVIDIA GPU (bitsandbytes)
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from data import load_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["lora", "dora", "qlora"], default="lora")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_steps", type=int, default=-1, help="override epochs, e.g. for a smoke test")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def pick_dtypes():
    """Returns (weight dtype, mixed-precision dtype or None)."""
    if not torch.cuda.is_available():
        return torch.float32, None  # MPS/CPU: only used for smoke tests
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, torch.bfloat16
    # T4 and older: keep weights in fp32 and only autocast to fp16, since pure fp16 Qwen can overflow to NaN.
    return torch.float32, torch.float16


def load_model(args, dtype, amp_dtype):
    kwargs = {"dtype": dtype}
    if args.method == "qlora":
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA needs an NVIDIA GPU (bitsandbytes). Run it on Colab/Kaggle.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=amp_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        use_dora=args.method == "dora",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main():
    args = parse_args()
    run_name = f"{args.method}-r{args.rank}"
    out = Path(args.out_dir) / run_name
    dtype, amp_dtype = pick_dtypes()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, val, test = load_splits(tokenizer, args.max_len, max_train_samples=args.max_train_samples,
                                   seed=args.seed)
    print(f"examples: train={len(train)} val={len(val)} test={len(test)}")

    model = load_model(args, dtype, amp_dtype)
    trainable, total = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float < 1 is a fraction of total steps
        bf16=amp_dtype == torch.bfloat16,
        fp16=amp_dtype == torch.float16,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60

    test_loss = trainer.evaluate(test, metric_key_prefix="test")["test_loss"]
    model.save_pretrained(out / "adapter")

    metrics = {
        "method": args.method,
        "model": args.model,
        "rank": args.rank,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_examples": len(train),
        "train_minutes": round(train_minutes, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(math.exp(test_loss), 2),
        "adapter_mb": round(dir_size_mb(out / "adapter"), 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": f"weights {dtype}, amp {amp_dtype}",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
