# Hierarchical TextNCA

Training and evaluation code for **Hierarchical TextNCA**, a causal language model built from the Neural Cellular Automaton (NCA) primitive: a local windowed-attention rule with a gated, weight-shared update that is applied repeatedly to the hidden state.

The model runs in stages with growing attention windows. Each stage has its own cell, iterated `T_s` times with shared weights, and the hidden state flows from one stage to the next without reset:

```
h = TokenEmbed(x) + PosEmbed(x)
for stage, w in enumerate([8, 32, 128]):          # narrow-to-wide window schedule
    for t in range(T_s):                          # T_s = 4 iterations per stage
        u = FFN(LayerNorm(LocalAttn_w(h + step_emb[t])))
        h = GRUGate(h, u)                         # gated in-place update
logits = LMHead(LayerNorm(h))
```

The flagship configuration (`[8, 32, 128] × T_s=4`, 12 NCA steps, d=512) has **30.8M parameters** and reaches **60.3 perplexity** on the WikiText-103 test set after training on 500M SlimPajama tokens.

## Repository layout

```
textnca/
  models/
    hier_nca.py      # HierarchicalTextNCA2 (final model) + stochastic-T_s variant
    components.py    # local attention kernels, gates, FFN, step embeddings
    config.py        # ModelConfig
    build.py         # builds the model from CLI / saved arguments
  data/
    language_modeling.py   # SlimPajama + WikiText-103 tokenization and loaders
  training/
    trainer.py       # training loop (AMP, grad accumulation, checkpointing)
    scheduler.py     # linear warmup + cosine decay
  utils.py
scripts/
  prepare_data.py    # tokenize SlimPajama (train) and WikiText-103 (test)
  train.py           # train a model
  evaluate.py        # WikiText-103 test perplexity from a checkpoint
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Experiments used PyTorch 2.11 with CUDA 13 on NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB); training runs in bf16.

## 1. Prepare data

```bash
python scripts/prepare_data.py --data_dir data/lm --num_tokens 500000000
```

This streams SlimPajama, tokenizes it with the Mistral-7B tokenizer (32k vocabulary), and writes `data/lm/slimpajama_500M_seq512.npy`. It also tokenizes the WikiText-103 test split into `data/lm/wikitext103_test_seq512.npy`. The Mistral tokenizer is gated on Hugging Face, so accept its terms and run `huggingface-cli login` first.

## 2. Train

The defaults are the flagship configuration:

```bash
python scripts/train.py --data_dir data/lm --total_steps 60000 --batch_size 32 --grad_accum 2 --seed 42
```

Outputs go to `results/<run_name>/`: `config.json`, `final_results.json`, and `checkpoints/` (`checkpoint_best.pt` holds the best WikiText-103 evaluation).

**Step counting:** `--total_steps` counts micro-batches. With `--grad_accum 2`, 60k steps are 30k optimizer updates of 32 × 2 × 512 = 32,768 tokens. The learning-rate scheduler advances once per optimizer update but is configured with `total_steps`, so the cosine decay only covers about half its curve by the end of training. This is the schedule that produced the paper's results, and it is kept unchanged so they can be reproduced.

## 3. Evaluate

```bash
python scripts/evaluate.py --run_dir results/<run_name> --data_dir data/lm
```

This rebuilds the model from `config.json`, loads `checkpoint_best.pt`, and reports perplexity over the full WikiText-103 test set (624 sequences of 512 tokens). Use `--eval_t_per_stage 2 4 6` to run the same checkpoint with a different number of iterations per stage.

## Ablations

All variants are selected with architecture flags on `scripts/train.py`. Each command below shows only the flags that differ from the flagship. Every run used `--total_steps 60000 --batch_size 32 --lr 6e-4`. The `grad_accum` column gives the value each paper run used.

| Variant | Flags | grad_accum | Params | WikiText-103 PPL |
|---|---|---|---|---|
| **Flagship** `[8,32,128] × T_s=4` | *(defaults)* | 2 | 30.8M | **60.3** |
| Reversed schedule | `--window_schedule 128 32 8` | 4 | 30.8M | 131.1 |
| Uniform schedule | `--window_schedule 32 32 32` | 4 | 30.8M | 119.7 |
| Wide-first schedule | `--window_schedule 32 8 128` | 4 | 30.8M | 82.2 |
| Non-monotone schedule | `--window_schedule 8 128 32` | 4 | 30.8M | 77.0 |
| K=2 stages | `--window_schedule 16 128 --t_per_stage 6` | 4 | 26.1M | 75.8 |
| K=4 stages | `--window_schedule 4 16 64 128 --t_per_stage 3` | 4 | 35.6M | 71.8 |
| Residual update | `--gate_type residual` | 4 | 26.1M | 76.6 |
| Sigmoid + dropout gate | `--gate_type sigmoid_dropout` | 2 | 27.7M | 71.7 |
| No step embedding | `--step_embed_type none` | 4 | 30.8M | 82.7* |
| Sinusoidal step embedding | `--step_embed_type sinusoidal` | 4 | 30.8M | 119.5 |
| GLA perception kernel | `--kernel_type local_gla` | 2 | 30.9M | 72.0 |
| T_s = 2 | `--t_per_stage 2` | 4 | 30.8M | 80.3 |
| T_s = 6 | `--t_per_stage 6` | 4 | 30.9M | 73.9 |
| T_s = 8 | `--t_per_stage 8` | 4 | 30.9M | 117.9 |
| Random T_s ∈ {2,4,6} | `--stochastic_t_range 2 4 6 --step_embed_type sinusoidal` | 4 | 30.8M | 101.1 |

\*Training was interrupted; the best checkpoint is from step 48k.

**Variance:** results vary substantially between runs. The paper reports flagship seeds 42 / 1337 / 2024 at 60.3 / 72.3 / 76.2 PPL, with mean 69.6 ± 8.3. The 1337 and 2024 runs used `--grad_accum 4`. A later rerun of the exact flagship command (seed 42, `--grad_accum 2`) reached 72.2.
