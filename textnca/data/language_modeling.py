"""Language modeling data: SlimPajama streaming, tokenization, and eval set loading.

Training: SlimPajama (Cerebras, 2023) — streamed from HuggingFace.
Evaluation: WikiText-103 test set.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─── Tokenized Memory-Mapped Dataset ────────────────────────────────────────

class MemmapLMDataset(Dataset):
    """Language modeling dataset backed by a memory-mapped numpy array.

    Stores pre-tokenized, pre-chunked sequences for efficient loading.
    """

    def __init__(self, data_path: str, seq_len: int = 512):
        self.data = np.load(data_path, mmap_mode="r")
        self.seq_len = seq_len
        # data shape: [num_sequences, seq_len+1] (input + next-token label)
        assert self.data.shape[1] >= seq_len + 1, \
            f"Data chunks must be at least seq_len+1={seq_len+1}, got {self.data.shape[1]}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx], dtype=torch.long)
        inputs = chunk[:self.seq_len]
        labels = chunk[1:self.seq_len + 1]
        return inputs, labels


# ─── SlimPajama Download and Tokenization ────────────────────────────────────

def prepare_slimpajama(
    output_dir: str,
    num_tokens: int = 2_000_000_000,
    seq_len: int = 512,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    seed: int = 42,
) -> str:
    """Stream SlimPajama, tokenize, chunk, and save as memory-mapped numpy array.

    Args:
        output_dir: Where to save the processed data
        num_tokens: Number of tokens to download (default 2B)
        seq_len: Sequence length for chunking
        tokenizer_name: HuggingFace tokenizer name
        seed: Random seed for shuffling

    Returns:
        Path to the saved .npy file
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install: pip install datasets transformers")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"slimpajama_{num_tokens // 1_000_000}M_seq{seq_len}.npy"

    if output_path.exists():
        print(f"Tokenized data already exists at {output_path}")
        return str(output_path)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id or 0

    print(f"Streaming SlimPajama, targeting {num_tokens:,} tokens...")
    try:
        ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    except Exception:
        print("  cerebras/SlimPajama-627B not available, using DKYoon/SlimPajama-6B")
        ds = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    all_tokens = []
    total_tokens = 0
    batch_texts = []
    batch_size = 1000

    for example in ds:
        batch_texts.append(example["text"])

        if len(batch_texts) >= batch_size:
            encoded = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
            for doc_ids in encoded:
                all_tokens.extend(doc_ids)
                all_tokens.append(tokenizer.eos_token_id)
                total_tokens += len(doc_ids) + 1

                if total_tokens >= num_tokens:
                    break
            batch_texts = []

            if total_tokens % 10_000_000 < batch_size * 100:
                print(f"  Processed {total_tokens:,} / {num_tokens:,} tokens ({100*total_tokens/num_tokens:.1f}%)")

            if total_tokens >= num_tokens:
                break

    # Process remaining
    if batch_texts and total_tokens < num_tokens:
        encoded = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        for doc_ids in encoded:
            all_tokens.extend(doc_ids)
            all_tokens.append(tokenizer.eos_token_id)

    # Chunk into sequences of seq_len + 1
    all_tokens = np.array(all_tokens[:num_tokens + seq_len], dtype=np.int32)
    num_chunks = len(all_tokens) // (seq_len + 1)
    data = all_tokens[:num_chunks * (seq_len + 1)].reshape(num_chunks, seq_len + 1)

    # Shuffle chunks
    rng = np.random.RandomState(seed)
    rng.shuffle(data)

    # Save
    np.save(str(output_path), data)
    print(f"Saved {num_chunks:,} chunks ({num_chunks * seq_len:,} tokens) to {output_path}")

    # Save metadata
    meta_path = output_dir / f"slimpajama_{num_tokens // 1_000_000}M_meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"tokenizer: {tokenizer_name}\n")
        f.write(f"vocab_size: {tokenizer.vocab_size}\n")
        f.write(f"seq_len: {seq_len}\n")
        f.write(f"num_chunks: {num_chunks}\n")
        f.write(f"total_tokens: {num_chunks * seq_len}\n")
        f.write(f"seed: {seed}\n")

    return str(output_path)


# ─── WikiText-103 Test Set ───────────────────────────────────────────────────

def prepare_wikitext103_test(
    output_dir: str,
    seq_len: int = 512,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
) -> str:
    """Download and tokenize WikiText-103 test set."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install: pip install datasets transformers")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"wikitext103_test_seq{seq_len}.npy"

    if output_path.exists():
        print(f"WikiText-103 test data already exists at {output_path}")
        return str(output_path)

    print("Loading WikiText-103 test set...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    all_tokens = []
    for example in ds:
        text = example["text"].strip()
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.int32)
    num_chunks = len(all_tokens) // (seq_len + 1)
    data = all_tokens[:num_chunks * (seq_len + 1)].reshape(num_chunks, seq_len + 1)

    np.save(str(output_path), data)
    print(f"Saved WikiText-103 test: {num_chunks} chunks to {output_path}")
    return str(output_path)


# ─── Convenience DataLoader Factories ────────────────────────────────────────

def get_slimpajama_dataloader(
    data_path: str,
    seq_len: int = 512,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = MemmapLMDataset(data_path, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_eval_dataloader(
    data_path: str,
    seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    ds = MemmapLMDataset(data_path, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
