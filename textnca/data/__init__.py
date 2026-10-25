"""Data module: SlimPajama training data and WikiText-103 evaluation data."""

from textnca.data.language_modeling import (
    MemmapLMDataset, prepare_slimpajama, prepare_wikitext103_test,
    get_slimpajama_dataloader, get_eval_dataloader,
)
