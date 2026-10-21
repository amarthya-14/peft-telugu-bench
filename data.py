"""Load the Telugu split of the Aya dataset and tokenize it for causal-LM fine-tuning."""

from datasets import load_dataset

DATASET = "CohereLabs/aya_dataset"  # human-written, Apache-2.0
LANG_CODE = "tel"


def _to_features(example, tokenizer, max_len):
    messages = [{"role": "user", "content": example["inputs"]}]
    # enable_thinking=False: Qwen3 otherwise expects a <think> block before the answer.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = prompt + example["targets"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    # Only the answer contributes to the loss; prompt tokens are masked with -100.
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {"input_ids": input_ids, "labels": labels, "keep": len(input_ids) <= max_len}


def _prepare(ds, tokenizer, max_len):
    ds = ds.filter(lambda r: r["language_code"] == LANG_CODE)
    ds = ds.map(_to_features, fn_kwargs={"tokenizer": tokenizer, "max_len": max_len},
                remove_columns=ds.column_names)
    # Drop over-long examples instead of truncating them, so no answer is cut mid-sentence.
    return ds.filter(lambda r: r["keep"]).remove_columns("keep")


def load_splits(tokenizer, max_len, val_size=300, max_train_samples=None, seed=42):
    """Returns (train, val, test). val is carved out of Aya train; test is Aya's own test split."""
    train = _prepare(load_dataset(DATASET, split="train"), tokenizer, max_len)
    test = _prepare(load_dataset(DATASET, split="test"), tokenizer, max_len)

    split = train.train_test_split(test_size=val_size, seed=seed)
    train, val = split["train"], split["test"]
    if max_train_samples:
        train = train.select(range(min(max_train_samples, len(train))))
        val = val.select(range(min(max_train_samples, len(val))))
        test = test.select(range(min(max_train_samples, len(test))))
    return train, val, test
