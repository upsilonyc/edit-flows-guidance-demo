import torch
import numpy as np
from ..main import *
from Levenshtein import distance

EPS = 1e-12
BOS_TOKEN = 128
PAD_TOKEN = 129
GAP_TOKEN = 130
# ===== util funcs for inference-time guidance =====
# 1. reward function

def trim_pad(seq, pad_token: int = PAD_TOKEN):
    if torch.is_tensor(seq):
        values = seq.detach().cpu().tolist()
    else:
        values = list(seq)
    return [int(v) for v in values if int(v) != pad_token]

def tokens_to_lev_string(tokens):
    # Shift by +1 to avoid null chars while preserving token identity mapping.
    return "".join(chr(int(t) + 1) for t in tokens)

# 2. sampling

def effective_sample_size(normalized_w: torch.Tensor) -> float:
    return float(1.0 / (normalized_w.pow(2).sum().item() + EPS))

# 3. reporting the results
def edit_distance_metrics(x, y):
    """Return raw and normalized Levenshtein distance for diagnostics."""
    x_tokens = trim_pad(x)
    y_tokens = trim_pad(y)
    x_str = tokens_to_lev_string(x_tokens)
    y_str = tokens_to_lev_string(y_tokens)
    d = float(distance(x_str, y_str))
    denom = float(max(len(x_str), len(y_str), 1))
    return d, d / denom

# ===== unused utils functions =====
# 1. generate target_y
def _make_target_sequence() -> torch.Tensor:
    _, x_1, _, _, _, _ = make_batch(
        batch_size=1,
        min_length=min_seq_len,
        max_length=max_seq_len,
        vocab_size=V,
        coupling=coupling,
        seq_align_fn=seq_align_fn,
        num_cycles_fn=num_cycles_fn,
        x_int_fn=x_int_fn,
    )
    return x_1[0].detach().cpu()