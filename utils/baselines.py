import torch
import numpy as np
# from main import min_seq_len, max_seq_len, make_batch, V, coupling, seq_align_fn, num_cycles_fn, x_int_fn
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

def generate_line_y(k=0.5, b=0, V = 128, noise_level=0.05, size=128):
    x = torch.arange(size, dtype=torch.float32)
    y = k * x + b
    y = (y - y.min()) / (y.max() - y.min() + EPS)
    y = y * V
    if noise_level > 0:
        noise = torch.randn_like(y) * noise_level  # independent noise per token
        y = y + noise
    return y

def generate_quadratic_y(a=1, b=1, c=0, V = 128, noise_level=0.05, size=128):
    x = torch.arange(size, dtype=torch.float32)
    y = a * x**2 + b * x + c
    y = (y - y.min()) / (y.max() - y.min() + EPS)
    y = y * V
    if noise_level > 0:
        y += noise_level * torch.randn_like(y)
    return y

def generate_sine_y(B=0.3, C=None, amplitude=0.5, offset=0.5, noise_level=0.05, size=128):
    # range in make_sinusoidal_sequence: B [0.75, 1.85] C [0, 2pi]
    if C is None:
        C = np.random.uniform(0,2*np.pi)
    x = torch.linspace(0, 4 * np.pi, size, dtype=torch.float32)
    y = amplitude * torch.sin(B * (x - C)) + offset
    if noise_level > 0:
        y += noise_level * torch.randn_like(y)
    return y

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
    denom = 256
    return d, d / denom

# ===== unused utils functions =====
# 1. generate target_y
# def make_target_sequence() -> torch.Tensor:
#     _, x_1, _, _, _, _ = make_batch(
#         batch_size=1,
#         min_length=min_seq_len,
#         max_length=max_seq_len,
#         vocab_size=V,
#         coupling=coupling,
#         seq_align_fn=seq_align_fn,
#         num_cycles_fn=num_cycles_fn,
#         x_int_fn=x_int_fn,
#     )
#     return x_1[0].detach().cpu()