import torch
from Levenshtein import distance
import numpy as np

BOS_TOKEN = 128
PAD_TOKEN = 129
GAP_TOKEN = 130
# ===== util funcs for inference-time guidance =====
# 1. reward function
def reward(x, y, beta=10): # (0, 1]
    d = distance(x, y)
    d = d / max(len(x), len(y))
    return np.exp(-beta * d)

def _trim_pad(seq, pad_token: int = PAD_TOKEN):
    if torch.is_tensor(seq):
        values = seq.detach().cpu().tolist()
    else:
        values = list(seq)
    return [int(v) for v in values if int(v) != pad_token]

def _tokens_to_lev_string(tokens):
    # Shift by +1 to avoid null chars while preserving token identity mapping.
    return "".join(chr(int(t) + 1) for t in tokens)

def edit_distance_reward(x, y, beta: int = 5) -> float:
    """Wrapper around the existing reward() function with robust token handling."""
    x_tokens = _trim_pad(x)
    y_tokens = _trim_pad(y)
    if len(x_tokens) == 0 and len(y_tokens) == 0:
        return 1.0
    x_str = _tokens_to_lev_string(x_tokens)
    y_str = _tokens_to_lev_string(y_tokens)
    return float(reward(x_str, y_str, beta=beta))

# ===== unused utils functions =====

