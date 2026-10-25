"""Prepare SlimPajama training data and WikiText-103 eval data.

Usage:
    python scripts/prepare_data.py --data_dir data/lm --num_tokens 500000000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textnca.data.language_modeling import prepare_slimpajama, prepare_wikitext103_test


def main():
    parser = argparse.ArgumentParser(description="Tokenize SlimPajama (train) and WikiText-103 (test)")
    parser.add_argument("--data_dir", type=str, default="data/lm")
    parser.add_argument("--num_tokens", type=int, default=500_000_000,
                        help="Number of SlimPajama training tokens (paper: 500M)")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Preparing SlimPajama ({args.num_tokens // 1_000_000}M tokens)...")
    print("=" * 60)
    train_path = prepare_slimpajama(
        output_dir=args.data_dir,
        num_tokens=args.num_tokens,
        seq_len=args.seq_len,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
    )
    print(f"Training data: {train_path}")

    print("\n" + "=" * 60)
    print("Preparing WikiText-103 test set...")
    print("=" * 60)
    eval_path = prepare_wikitext103_test(
        output_dir=args.data_dir,
        seq_len=args.seq_len,
        tokenizer_name=args.tokenizer,
    )
    print(f"Eval data: {eval_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
