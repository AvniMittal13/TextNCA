"""Evaluate a trained Hierarchical TextNCA checkpoint: WikiText-103 test perplexity.

Usage:
    # Rebuilds the architecture from <run_dir>/config.json
    python scripts/evaluate.py --run_dir results/<run_name>

    # Evaluate the same checkpoint at a different number of iterations per stage
    python scripts/evaluate.py --run_dir results/<run_name> --eval_t_per_stage 2 4 6

    # Checkpoint without an "arch" block in config.json: pass architecture flags instead
    python scripts/evaluate.py --checkpoint path/to/checkpoint_best.pt --window_schedule 8 32 128
"""
import argparse
import sys
import os
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from textnca.utils import set_seed, count_parameters
from textnca.models.build import add_arch_args, arch_from_args, build_model
from textnca.data.language_modeling import get_eval_dataloader


@torch.no_grad()
def evaluate_ppl(model, dataloader, device, t_per_stage=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    use_amp = device == "cuda"

    for batch in dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(inputs, num_steps=t_per_stage)["logits"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 100)), "tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical TextNCA on WikiText-103")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory written by scripts/train.py")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: <run_dir>/checkpoints/checkpoint_best.pt)")
    parser.add_argument("--data_dir", type=str, default="data/lm")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_t_per_stage", type=int, nargs="+", default=None,
                        help="Override T_s at inference (default: the trained T_s)")
    add_arch_args(parser)
    args = parser.parse_args()

    if args.run_dir is None and args.checkpoint is None:
        parser.error("pass --run_dir or --checkpoint")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    arch = arch_from_args(args)
    if args.run_dir:
        config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                arch = json.load(f).get("arch", arch)
    ckpt_path = args.checkpoint or os.path.join(args.run_dir, "checkpoints", "checkpoint_best.pt")

    model = build_model(arch, use_gradient_checkpointing=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded {ckpt_path} (step {ckpt.get('global_step')}, "
          f"{count_parameters(model)['total']:,} params)", flush=True)

    eval_path = os.path.join(args.data_dir, "wikitext103_test_seq512.npy")
    eval_dl = get_eval_dataloader(eval_path, seq_len=512, batch_size=args.batch_size)

    t_values = args.eval_t_per_stage or [arch["t_per_stage"]]
    results = {}
    print(f"\n{'T_s':>4} {'Total steps':>12} {'Loss':>10} {'PPL':>10}")
    print("-" * 40)
    for t_s in t_values:
        r = evaluate_ppl(model, eval_dl, device, t_per_stage=t_s)
        results[t_s] = {"t_per_stage": t_s, "total_steps": len(arch["window_schedule"]) * t_s, **r}
        print(f"{t_s:>4} {results[t_s]['total_steps']:>12} {r['loss']:>10.4f} {r['ppl']:>10.2f}", flush=True)

    if args.run_dir:
        out_path = os.path.join(args.run_dir, "eval_wikitext103.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
