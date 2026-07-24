# %% path to GPU
import os
import sys
true_cuda_path = "/usr/lib/x86_64-linux-gnu"
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if true_cuda_path not in current_ld:
    os.environ["LD_LIBRARY_PATH"] = true_cuda_path + (f":{current_ld}" if current_ld else "")

import torch
print("\n" + "="*10)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显卡:", torch.cuda.get_device_name(0))
print("="*10 + "\n")
# %% [markdown]
# # Edit Flows: Discrete Flow Matching with Edit Operations
# 
# Here, I replicate the **"Edit Flows: Flow Matching with Edit Operations"** paper by [Havasi et al.](https://arxiv.org/abs/2506.09018) which introduces a non-factorized probability path model for discrete flow matching via insertions, deletions, and substitutions.

# %% [markdown]
# ## Dataset Overview
# 
# For this educational implementation, I will use a synthetic dataset of discretized noisy sinusoidal sequences. The key variables that can be adjusted to create a variety of sequences include:
# 
# - **Sequence Length**: The number of time steps in each sequence.
# - **Number of Cycles**: The number of sinusoidal cycles in the sequence.
# - **Noise Level**: The amount of Gaussian noise added to the sinusoidal signal.
# - **X-Axis Offset**: A constant value to shift the x-axis of the sinusoidal function.
# 
# We can vary the distributions from which we sample these parameters to create a target distribution of varying modelling complexity. Further, our choice of the base distribution can also be adjusted to create more or less complex training setup. We will elaborate on this further in the training section.

# %%
# Function to generate sinusoidal sequence data with optional noise

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def make_sinusoidal_sequence( # generates x_1
    x_seq_len: int,
    noise: float,
    num_cycles_fn: Callable[[], float] = lambda: np.random.uniform(1.5, 3.5),
    x_int_fn: Callable[[], float] = lambda: np.random.uniform(0, 2*np.pi)
) -> np.ndarray:
    """
    Generate a discretized sinusoidal sequence with optional Gaussian noise.
    The sinusoidal function follows: y = 1/2 * sin(B(x-C)) + 1/2 where B and C are randomly chosen.
    """
    x = np.linspace(0, 4*np.pi, x_seq_len)
    num_cycles = num_cycles_fn()
    B = 2 * np.pi * num_cycles / (4 * np.pi) # num of cycles
    C = x_int_fn()
    y = 0.5 * np.sin(B * (x - C)) + 0.5    
    if noise > 0:
        gaussian_noise = np.random.normal(0, noise, x_seq_len)
        y += gaussian_noise
    return y

def plot_sequences(xs: np.ndarray, title: str = "Sequences", pad_token: int | None = None):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid()
    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0, 1, xs.shape[0])) # MODIFIED. original version uses virdis
    for i, y in enumerate(xs):
        if pad_token is not None:
            y = y[y != pad_token]
        x = np.arange(len(y))
        plt.scatter(x, y, label=f"Sequence {i+1}", s=10, c=colors[i])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Training Edit Flow Models
# 
# To train a regular discrete flow matching model, we sample pairs from a coupling distribution $\pi(x_0,x_1)$ that marginalizes to the base and target distributions. Then, we sample a time step $t \in [0, 1]$ and sample from the forward conditional probability path $x_t \sim p_t(x|x_0,x_1)$. Using these ingredients, we then approximate the marginal rate $u_t(x|x_t) = \mathbb{E}_{p_t(x_0,x_1|x_t)}\left[ u_t(x|x_t,x_0,x_1) \right]$ by minimizing a cross-entropy loss or maximizing an ELBO.
# 
# ### Edit Flows Training
# 
# Edit Flows are slightly more complex since there is no clear choice for the conditional probability path $p_t(x|x_0,x_1)$, as the edit operations dramatically expand the possible edit paths that can be taken from $x_0$ to $x_1$. Instead, the authors introduce an auxiliary variable $Z$ representing the space of **aligned** sequences after introducing a special *gap* token. Given a unique mapping $f: Z \to X$, we can simply define the conditional probability path over $Z$ to sample $z_t$, derive the training target in this space, then map the inputs back onto the original space $X$ to input to the model. In greater detail, the training procedure is as follows:
# 
# 1. Sample a pair $(x_0, x_1)$ from the coupling distribution $\pi(x_0, x_1)$
# 2. Align the sequence pair (e.g. Levenstein algorithm) to obtain $(z_0, z_1)$ of equal length
# 3. Sample a time step $t \in [0, 1]$
# 4. Sample a sequence $z_t$ from the conditional probability path $p_t(x,z|x_0,x_1,z_0,z_1)$
# 5. Map $z_t$ back to the original space $x_t = f(z_t)$
# 6. Pass $x_t$ through the model to obtain rates $u_{t,i}(x|x_t)$ for every possible edit operation
# 7. Apply the Bregman divergence loss to the predicted rates which minimizes all output rates of the model while having a weighted cross-entropy over edit operations that bring $x_t$ closer to $x_1$ in the $Z$ reference frame

# %% [markdown]
# ### Model Overview
# 
# Like previous flow matching setups, Edit Flows learns the instantaneous marginal rate that transports from the base distribution $p(x)$ to the target distribution $q(x)$. This means that we must train a neural network to learn this marginal rate $u_t(x|x_t)$ for all time steps $t \in [0, 1]$ and for all the possible states $x$ that are accessible from $x_t$ via a single edit operation (insertion, deletion, or substitution). Thus, we can easily define the input and outputs of the model as follows:
# 
# ![](static/editflows-fig-1.png)
# _Figure 1 taken from the Edit Flows paper_
# 
# - **Input**: A tokenized sequence $x_t$ and a time step $t$
# - **Output**: A vector of marginal rates for the edit operations for each position in the sequence
# 
# In effect, for each position in the sequence, we must predict:
# - $\lambda_{t,i}^{\text{ins}}(x_t)$: the marginal rate of inserting **any** token at position $i$
# - $\lambda_{t,i}^{\text{del}}(x_t)$: the marginal rate of deleting the token at position $i$
# - $\lambda_{t,i}^{\text{sub}}(x_t)$: the marginal rate of substituting the token at position $i$
# - $Q_{t,i}^{\text{ins}}(a|x_t)$: the probability of picking token $a$ if we **insert** at position $i$
# - $Q_{t,i}^{\text{sub}}(a|x_t)$: the probability of picking token $a$ if we **substitute** at position $i$
# 
# Given a vocabulary of size $V$, the model thus outputs a vector of size $3 + 2V$ for each position in the sequence, where the first three values correspond to the marginal rates of insertion, deletion, and substitution, and the remaining $2V$ values correspond to the probabilities of inserting or substituting each token in the vocabulary. If the sequence has length $L$, the model outputs a matrix of size $L \times (3 + 2V)$.

# %%
# Basic Transformer model architecture for edit flows

from typing import Tuple, cast
from torchtyping import TensorType as T

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Simple sinusoidal time embedding for Transformer model
    """
    def __init__(self, hidden_dim: int):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: T["batch", 1, "float"]) -> T["batch", "hidden_dim"]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # type: ignore (MODIFIED) # Make it (batch_size, 1)

        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)  # Broadcasting: (batch_size, 1) * (1, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd hidden_dim
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb  # type: ignore (MODIFIED) # (batch_size, hidden_dim)


class SimpleEditFlowsTransformer(nn.Module):
    """
    Small vanilla Transformer model for edit flows with padding support.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads = 8,
        max_seq_len=512,
        bos_token_id=128,
        pad_token_id=129,
    ):
        super(SimpleEditFlowsTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        assert bos_token_id < vocab_size, "bos_token_id must be less than vocab_size"
        assert pad_token_id < vocab_size, "pad_token_id must be less than vocab_size"

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)     # Token embeddings
        self.time_embedding = nn.Sequential(                            # Time embeddings
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)      # Positional embeddings
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                        dim_feedforward=hidden_dim * 4,
                                        dropout=0.1, activation='gelu',
                                        batch_first=False)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Output 3 rates (insert, substitute, delete)
        )
        self.ins_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),  # Output vocab_size insert probabilities
        )
        self.sub_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),  # Output vocab_size substitute probabilities
        )
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: T["batch", "x_seq_len", "long"], 
                time_step: T["batch", 1, "float"],
                padding_mask: T["batch", "x_seq_len", "bool"]) \
        -> Tuple[
            T["batch", "x_seq_len", "float"],         # Rates (3 values)
            T["batch", "x_seq_len", "vocab_size"],    # Insert probabilities (vocab_size values)
            T["batch", "x_seq_len", "vocab_size"],    # Substitute probabilities (vocab_size values)
        ]:
        """Forward pass takes in x_t, t, and padding mask, returns rates and probabilities
        """
        batch_size, x_seq_len = tokens.shape        
        token_emb = self.token_embedding(tokens)    # (batch_size, x_seq_len, hidden_dim)        

        time_emb = self.time_embedding(time_step)   # (batch_size, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, x_seq_len, -1)  # (batch_size, x_seq_len, hidden_dim)

        positions = torch.arange(x_seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)     # (batch_size, x_seq_len, hidden_dim)
        
        x = token_emb + time_emb + pos_emb          # (batch_size, x_seq_len, hidden_dim)
        x = x.transpose(0, 1)                       # expects (x_seq_len, batch_size, hidden_dim)   
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = x.transpose(0, 1)                           # (batch_size, x_seq_len, hidden_dim)
        x = self.final_layer_norm(x)                    # (batch_size, x_seq_len, hidden_dim)
        ins_logits = self.ins_logits_out(x)             # (batch_size, x_seq_len, vocab_size)
        sub_logits = self.sub_logits_out(x)             # (batch_size, x_seq_len, vocab_size)
        rates = F.softplus(self.rates_out(x))           # (batch_size, x_seq_len, 3) - ensure positive rates

        ins_probs = F.softmax(ins_logits, dim=-1)   # (batch_size, x_seq_len, vocab_size)
        sub_probs = F.softmax(sub_logits, dim=-1)   # (batch_size, x_seq_len, vocab_size)
        
        # Zero out outputs for padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, x_seq_len, 1)
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded
        
        if torch.isnan(rates).any() or torch.isnan(ins_probs).any() or torch.isnan(sub_probs).any():
            raise ValueError("NaN detected in output probabilities or rates")

        return (
            cast(T["batch", "x_seq_len", "float"], rates),
            cast(T["batch", "x_seq_len", "vocab_size"], ins_probs),
            cast(T["batch", "x_seq_len", "vocab_size"], sub_probs),
        )

# %%
# Setup the model and optimizer
V = 128 # vocab size
L = 128 # max sequence length

torch.manual_seed(42)
np.random.seed(42)

model = SimpleEditFlowsTransformer(
    vocab_size=V+2,  # +2 for PAD + BOS tokens
    hidden_dim=512,
    num_layers=8,
    num_heads=32,
    max_seq_len=2*L,
    pad_token_id=V,
    bos_token_id=V+1,
)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# Print some model statistics and details
print(f"Model: {model.__class__.__name__}")
print(f"  Vocab size: {model.vocab_size}")
print(f"  Hidden dim: {model.hidden_dim}")
print(f"  Num layers: {model.num_layers}")
print(f"  Max seq len: {model.max_seq_len}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Print some details about the optimizer
print(f"Optimizer: {optim.__class__.__name__}")
print(f"  Learning rate: {optim.defaults['lr']}")

# %% [markdown]
# ### Training Objective
# Here we implement the training loop for Edit Flows.
# 
# ![loss](static/editflows-loss.png)
# _Equation 23 taken from the Edit Flows paper_

# %%
# Helper functions for training the model

from utils.utils import * # noqa
from flow import KappaScheduler, Coupling, EmptyCoupling, sample_p

def sample_cond_pt(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, kappa: KappaScheduler):
    t = t.reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt) # construct x_t

def make_x0_like_x1(
    x1: torch.Tensor,
    vocab_size: int = 128,
    pad_token: int = PAD_TOKEN,
    noise: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    batch_size, x1_max_len = x1.shape
    x0s = []
    for i in range(batch_size):
        x1i_len = (x1[i] != pad_token).sum().item()
        x0i = torch.Tensor(make_sinusoidal_sequence(int(x1i_len), noise=noise, **kwargs))
        x0i = torch.round(torch.clip(x0i * vocab_size, min=0.0, max=vocab_size-1)).long()
        x0i = F.pad(x0i, (0, x1_max_len - x0i.shape[0]), value=pad_token)
        x0s.append(x0i)
    x0s = torch.stack(x0s, dim=0).long()  # (batch_size, x1_max_len)
    assert x0s.shape == x1.shape, "x0 and x1 must have the same shape"
    return x0s

def make_x0_with_bounds(
    batch_size: int = 2,
    min_length: int = 96,
    max_length: int = 96,
    vocab_size: int = 128,
    pad_token: int = PAD_TOKEN,
    noise: float = 0.05,
    **kwargs
) -> torch.Tensor:
    lengths = np.random.randint(min_length, max_length+1, size=(batch_size,))
    max_seq_len = lengths.max()
    x0s = []
    for length in lengths:
        x0i = torch.Tensor(make_sinusoidal_sequence(length, noise=noise, **kwargs))
        x0i = torch.round(torch.clip(x0i * vocab_size, min=0.0, max=vocab_size-1)).long()
        x0i = F.pad(x0i, (0, max_seq_len - x0i.shape[0]), value=pad_token)
        x0s.append(x0i)
    x0s = torch.stack(x0s, dim=0).long()  # (batch_size, max_seq_len)
    assert x0s.shape[1] == max_seq_len
    assert x0s.shape[0] == batch_size
    return x0s

def make_batch(
    batch_size: int = 2,
    min_length: int = 96,
    max_length: int = 96,
    vocab_size: int = 128,
    pad_token: int = PAD_TOKEN,
    bos_token: int = BOS_TOKEN,
    coupling: Coupling = EmptyCoupling(),
    seq_align_fn = opt_align_xs_to_zs,
    noise: float = 0.05,
    **kwargs,
):
    lengths = np.random.randint(min_length, max_length+1, size=batch_size)
    x_1, x_0 = [], []
    z_1, z_0 = [], []

    for length in lengths:
        _x1 = torch.Tensor(make_sinusoidal_sequence(length, noise=noise, **kwargs))
        _x1 = torch.round(torch.clip(_x1 * vocab_size, min=0.0, max=vocab_size-1)).long().unsqueeze(0)
        _x0, _ = coupling.sample(_x1)
        _z0, _z1 = seq_align_fn(_x0, _x1)
        x_1.append(_x1.squeeze(0))
        x_0.append(_x0.squeeze(0))
        z_1.append(_z1.squeeze(0))
        z_0.append(_z0.squeeze(0))

    # Find the maximum length of each sequence in the batch
    x0_max_len = max(len(x) for x in x_0)
    x1_max_len = max(len(x) for x in x_1)
    z_max_len = max(len(z) for z in z_1)
    assert z_max_len == max(len(z) for z in z_0), "z_1 and z_0 must have the same max length"

    # Add <PAD> token at end of each sequence to make them equal length
    x_1 = torch.stack([F.pad(x, (0, x1_max_len - x.shape[0]), value=pad_token) for x in x_1], dim=0).long()
    x_0 = torch.stack([F.pad(x, (0, x0_max_len - x.shape[0]), value=pad_token) for x in x_0], dim=0).long()
    z_1 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_1], dim=0).long()
    z_0 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_0], dim=0).long()

    # Add <BOS> token at the start of each sequence
    x_1 = F.pad(x_1, (1, 0), value=bos_token)
    x_0 = F.pad(x_0, (1, 0), value=bos_token)
    z_1 = F.pad(z_1, (1, 0), value=bos_token)
    z_0 = F.pad(z_0, (1, 0), value=bos_token)

    t = torch.rand(batch_size, 1)
    padding_mask = (x_1 == pad_token)
    return x_0, x_1, z_0, z_1, t, padding_mask

def make_ut_mask_from_z(
    z_t: T["batch_size", "z_seq_len", "long"],
    z_1: T["batch_size", "z_seq_len", "long"],
    vocab_size: int = 130,
    pad_token: int = PAD_TOKEN,
    gap_token: int = GAP_TOKEN,
) -> T["batch_size", "z_seq_len", "n_ops", "bool"]:
    """
    Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
    For each position i where z_t and z_1 differ, we index as follows:

    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert, c] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1  # insert + substitute + delete

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq         # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq         # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del                                 # (batch_size, z_seq_len) 

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:,:,-1][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return cast(T["batch_size", "z_seq_len", "n_ops", "bool"], u_mask)

def fill_gap_tokens_with_repeats(
    x_ut: torch.Tensor,
    z_gap_mask: torch.Tensor,
    z_pad_mask: torch.Tensor,
):
    batch_size, _ = z_gap_mask.shape
    _, x_seq_len, _ = x_ut.shape

    # Use cumsum on non-gap positions to point to the last valid non-gap position
    non_gap_mask = ~z_gap_mask  # Invert mask to get non-gap positions
    indices = non_gap_mask.cumsum(dim=1) - 1        # (batch_size, z_seq_len)
    indices = indices.clamp(min=0, max=x_seq_len-1) # Ensure indices are within bounds

    # Use indices to gather from x_ut
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]   # (batch_size, z_seq_len, vocab_size)
    result[z_pad_mask] = 0                  # Set pad positions to 0
    return result

# %%
# Training loop
# 
import sys
import hashlib
from tqdm import tqdm
from collections import defaultdict
from flow import CubicScheduler, EmptyCoupling, GeneratorCoupling, ExtendedCoupling, UniformCoupling, x2prob
from IPython.display import clear_output

debug = False
metrics = defaultdict(list)
batch_size = 1 # MODIFIED to avoid OOM (original: 128, 64, 128)
min_seq_len = 64 # revert to original if GPU is available.
max_seq_len = 128

seq_align_fn = opt_align_xs_to_zs
# seq_align_fn = shifted_align_xs_to_zs

num_cycles_fn = lambda: np.random.uniform(2.5, 4)
# num_cycles_fn = lambda: 3.5
x_int_fn = lambda: np.random.uniform(0, 2*np.pi)

generator_fn = lambda x1: make_x0_with_bounds(batch_size=int(x1.shape[0]), min_length=min_seq_len, max_length=max_seq_len,
                                              vocab_size=V, pad_token=PAD_TOKEN, num_cycles_fn=lambda: np.random.uniform(1., 2.5), x_int_fn=x_int_fn)
# generator_fn = lambda x1: make_x0_like_x1(
#     x1, vocab_size=V, pad_token=PAD_TOKEN, num_cycles_fn=lambda: np.random.uniform(1, 2.5), x_int_fn=x_int_fn)

# coupling = EmptyCoupling()
coupling = GeneratorCoupling(generator_fn=generator_fn)
# coupling = ExtendedCoupling(n_insert=64, vocab_size=V, pad_token=PAD_TOKEN)
# coupling = UniformCoupling(
#     min_len=min_seq_len, max_len=max_seq_len, mirror_len=True, vocab_size=V, pad_token=PAD_TOKEN)

scheduler = CubicScheduler(a=1.0, b=1.0)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device)
model.train()

steps = 384000 // batch_size
pbar = tqdm(range(steps), desc="Training Edit Flows", unit="step")

# %%
# Helper functions for sampling the model

def apply_ins_del_operations(
    x_t: T["batch_size", "seq_len", "long"],
    ins_mask: T["batch_size", "seq_len", "bool"],
    del_mask: T["batch_size", "seq_len", "bool"],
    ins_tokens: T["batch_size", "seq_len", "long"],
    max_seq_len: int = 512,
    pad_token=GAP_TOKEN,
) -> T["batch_size", "max_new_len", "long"]:
    """
    Apply insertion and deletion operations to a sequence x_t based on the provided masks.
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substitutions
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    # Update ins/del masks after handling replacements
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    # Compute new lengths after applying ins/del operations
    xt_pad_mask = (x_t == pad_token)  # (batch_size, seq_len)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)  # (batch_size,)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
        return cast(
            T["batch_size", "max_new_len", "long"],
            torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device),
        )

    # Pre-allocate result
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)

    # Compute positions
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)    # (batch_size, 1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)         # (1, seq_len)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)                 # num del up to & incl. current pos
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)                 # num ins up to & incl. current pos
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)            # num ins before current pos
    
    # Place non-deleted tokens
    new_pos = pos_idx + cum_ins_before - cum_del                            # new pos of tokens shifted by ins/del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)    # tokens to keep (non-deleted)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

    # Place insertions
    if eff_ins_mask.any():
        ins_pos = new_pos + 1                                               # insertions go 1 after new shifted pos
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len) # tokens to insert
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]
    
    if max_new_len > max_seq_len:
        # print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
        max_new_len = max_seq_len
    
    return cast(
        T["batch_size", "max_new_len", "long"],
        x_new[:, :max_new_len],
    )

def get_adaptive_h(h: float, t: torch.Tensor, scheduler: KappaScheduler):
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    _h = h * torch.ones_like(t, device=t.device)
    h_adapt = torch.minimum(_h, coeff)
    return h_adapt

def load_model_state(model_name: str):
    checkpoint = torch.load(model_name, map_location=device)
    model = SimpleEditFlowsTransformer(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        max_seq_len=checkpoint['max_seq_len'],
        bos_token_id=checkpoint['bos_token_id'],
        pad_token_id=checkpoint['pad_token_id'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    return model.to(device), optim

# %%
# (Optional) Save / Load the model state
from pathlib import Path

save_model = False
load_model = True
overwrite = True

# save_dir = Path(f"results")
model_name = Path(f"checkpoint.pt") # MODIFIED. previous: seq2seq_prior.pt

if save_model:
    save_path = model_name
    if not overwrite:
        assert not save_path.exists(), f"Model file {save_path} already exists. Please choose a different name."
    assert save_path.parent.exists(), f"Directory {save_path.parent} does not exist. Please create it first."
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'vocab_size': model.vocab_size,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'max_seq_len': model.max_seq_len,
        'bos_token_id': model.bos_token_id,
        'pad_token_id': model.pad_token_id,
    }, save_path)
    print(f"Model saved to {save_path}")

if load_model:
    save_path = model_name
    assert save_path.exists(), f"Model file {save_path} does not exist."
    model, optim = load_model_state(str(save_path))
    print(f"Model loaded from {save_path}")

# %% [markdown]
# # Inference

# %% [markdown]
# ### Reward Function:
# Below I implement two alternatives: 
# - Using Levensthein Distance: The reward function R gives the similarity between a generated sequence x and the desired sequence y. The reward function is baised on the Leveinshtein distance (d) between the generated sequence, x, and the target sequence, y: $d_{norm}​=\frac{d(x,y)}{L}​$, and the reward $R = exp(−βd_{norm}​)$, where the exponential term makes the reward always non-negative, which is better for adjusting weights in SMC. 
# - Using Pearson correlaton between dynamic time-warped sequences: Compared to the previous reward function, this reflects the quantitative distances between x and y, and account for phase shifts. x and y are first aligned using Dynamic Time Warping (DTW) to wx and wy, which are of equal length, then we compute the Pearson correlation between x and y to normalize the reward. 

# %%
# reward func using Levenshtein distance
from Levenshtein import distance
from utils.baselines import *
from tslearn.metrics import dtw_path

lev = True

def levenshtein(x, y, alpha=5): # (0, 1]
    d = distance(x, y)
    d = d / L
    return np.exp(-alpha * d)

def dtw(x, y, alpha):
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    path, d = dtw_path(x_arr, y_arr)
    # path_a, path_b = zip(*path)
    # wx = x_arr[list(path_a)]
    # wy = y_arr[list(path_b)]
    # # Correlation is undefined for constant sequences; treat as neutral reward.
    # if np.std(wx) < EPS or np.std(wy) < EPS:
    #     return EPS
    # corr = np.corrcoef(wx, wy)[0, 1]
    d = d / L
    return np.exp(-alpha * d)

reward = None
if lev:
    reward = levenshtein
else:
    reward = dtw
    
def edit_distance_reward(x, y, alpha: int = 5) -> float:
    """Wrapper around the existing reward() function with robust token handling."""
    x_tokens = trim_pad(x)
    y_tokens = trim_pad(y)
    if len(x_tokens) == 0 and len(y_tokens) == 0:
        return 1.0

    if lev:
        x_str = tokens_to_lev_string(x_tokens)
        y_str = tokens_to_lev_string(y_tokens)
        return float(reward(x_str, y_str, alpha=alpha)) # type: ignore

    x_arr = np.asarray(x_tokens, dtype=np.float32)
    y_arr = np.asarray(y_tokens, dtype=np.float32)
    if x_arr.size == 0 or y_arr.size == 0:
        return EPS
    return float(reward(x_arr, y_arr, alpha=alpha)) # type: ignore

# %% [markdown]
# ### Guidance
# ##### Exact Guidance on $\lambda$
# Using the reward, r, as the predictor in exact guidance, the exact guidance goes as follows: 
# $$\tilde{R}_t​(x,\tilde{x})=R_t​(x,\tilde{x})\frac{r(\tilde{x},y)}{r(x,y)​}$$
# where $\tilde{x}$ is produced by $f(x, a)$ for any possible edit $a$. 
# 
# Here we don't directly modify the rate matrix Q used by ins and sub. Instead, exact guidance only applies to the three lambda edit rates: $$λ^*_{sub}​=λ_{sub}​⋅E_{a∼p_{sub}​​}[\frac{r(x′_a​)}{r(x)}]$$
# ##### Exact Guidance on Rates u
# Compute $u$ from $\lambda$-s and $Q$, and make guidance on u. 
# Specifically, $u$-s are obtained by: 
# <img src="static/editflows-fig-u.png" alt="description" style="max-width: 80%; box-sizing: border-box;">
# 
# So $u_{i, a} = \lambda_i \cdot Q_i(a)$

# %%
# Inference-time utilities, reusable sampler, and guidance baselines

import time
from utils.baselines import *
from typing import Callable, Dict, Optional
import pandas as pd
import json

def make_target_pair():
    """Return (x0, x1) from the same coupling sample in make_batch()."""
    x0, x1, _, _, _, _ = make_batch(
        batch_size=1,
        min_length=min_seq_len,
        max_length=max_seq_len,
        vocab_size=V,
        coupling=coupling,
        seq_align_fn=seq_align_fn,
        num_cycles_fn=num_cycles_fn,
        x_int_fn=x_int_fn,
    )
    return x0[0].detach().cpu(), x1[0].detach().cpu()

# paired_x0 = make_x0_with_bounds(batch_size=batch_size, min_length=min_seq_len, max_length=max_seq_len)[0]
paired_x0, _, _, _, _, _ = make_batch(
        batch_size=1,
        min_length=min_seq_len,
        max_length=max_seq_len,
        vocab_size=V,
        coupling=coupling,
        seq_align_fn=seq_align_fn,
        num_cycles_fn=num_cycles_fn,
        x_int_fn=x_int_fn,
    )
paired_x0 = paired_x0[0]
target_y = generate_quadratic_y()
# paired_x0, target_y = make_target_pair()

def sequence_logprob(step_logprobs: list[torch.Tensor]) -> torch.Tensor:
    """Return trajectory log-likelihood by summing per-step log-probabilities."""
    if len(step_logprobs) == 0:
        return torch.zeros(1, dtype=torch.float32)
    return torch.stack(step_logprobs, dim=0).sum(dim=0)

def _virtual_apply_edit(tokens, op: str, pos: int, token: Optional[int] = None):
    new_tokens = list(tokens)
    if op == "ins":
        assert token is not None
        insert_pos = min(pos + 1, len(new_tokens))
        new_tokens.insert(insert_pos, int(token))
    elif op == "sub":
        if 0 <= pos < len(new_tokens):
            assert token is not None
            new_tokens[pos] = int(token)
    elif op == "del":
        if 0 <= pos < len(new_tokens):
            del new_tokens[pos]
    return new_tokens

def _ctmc_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    default_h: float,
    guidance: Optional[Callable] = None,
    target_y: Optional[torch.Tensor] = None,
    return_logprob: bool = False,
    beta: int = 5
):
    """Single Euler-CTMC step shared by all methods."""
    x_t = x_t.detach().cpu()
    x_pad_mask = (x_t == PAD_TOKEN)

    with torch.no_grad():
        u_t, ins_probs, sub_probs = model.forward(
            cast(T["batch_size", "x_seq_len", "long"], x_t.to(device)),
            cast(T["batch_size", 1, "float"], t.to(device)),
            cast(T["batch_size", "x_seq_len", "bool"], x_pad_mask.to(device)),
        )

    lambda_ins = u_t[:, :, 0].detach().cpu()
    lambda_sub = u_t[:, :, 1].detach().cpu()
    lambda_del = u_t[:, :, 2].detach().cpu()
    ins_probs = ins_probs.detach().cpu()
    sub_probs = sub_probs.detach().cpu()

    if guidance is not None:
        guided = guidance(
            x_t,
            lambda_ins,
            lambda_sub,
            lambda_del,
            ins_probs,
            sub_probs,
            target_y,
            beta
        )
        if isinstance(guided, (tuple, list)) and len(guided) == 3: # exact_guidance_lambda
            lambda_ins, lambda_sub, lambda_del = guided
        elif isinstance(guided, (tuple, list)) and len(guided) == 5:
            lambda_ins, lambda_sub, lambda_del, ins_probs, sub_probs = guided
        else:
            raise ValueError("Guidance must return 3-tuple (lambdas) or 5-tuple (lambdas + probs).")

    adapt_h = get_adaptive_h(default_h, t.cpu(), scheduler)

    # Bernoulli thinning for independent insertion and combined delete/substitute events.
    p_ins = (1 - torch.exp(-adapt_h * lambda_ins)).clamp(min=0.0, max=1.0)
    p_del_sub = (1 - torch.exp(-adapt_h * (lambda_sub + lambda_del))).clamp(min=0.0, max=1.0) # del and sub are exclusive

    ins_mask = torch.rand_like(p_ins) < p_ins # determines positions with insertion operations
    del_sub_mask = torch.rand_like(p_del_sub) < p_del_sub

    rate_sum = (lambda_sub + lambda_del).clamp_min(EPS)
    prob_del = torch.where(del_sub_mask, lambda_del / rate_sum, torch.zeros_like(lambda_del)).clamp(min=0.0, max=1.0)
    del_mask = torch.bernoulli(prob_del).bool()
    sub_mask = del_sub_mask & ~del_mask

    ins_tokens = torch.full(ins_probs.shape[:2], PAD_TOKEN, dtype=torch.long)
    sub_tokens = torch.full(sub_probs.shape[:2], PAD_TOKEN, dtype=torch.long)

    non_pad_mask = ~x_pad_mask
    if non_pad_mask.any():
        ins_sampled = torch.multinomial(ins_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(-1)
        sub_sampled = torch.multinomial(sub_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(-1)
        ins_tokens[non_pad_mask] = ins_sampled
        sub_tokens[non_pad_mask] = sub_sampled

    x_next = x_t.clone()
    x_next[sub_mask] = sub_tokens[sub_mask]
    x_next = apply_ins_del_operations(
        cast(T["batch_size", "seq_len", "long"], x_next),
        cast(T["batch_size", "seq_len", "bool"], ins_mask),
        cast(T["batch_size", "seq_len", "bool"], del_mask),
        cast(T["batch_size", "seq_len", "long"], ins_tokens),
        max_seq_len=model.max_seq_len,
        pad_token=PAD_TOKEN,
    )

    step_logprob = None
    if return_logprob:
        active_mask = non_pad_mask.float()

        p_ins_safe = p_ins.clamp(min=EPS, max=1 - EPS)
        p_del_sub_safe = p_del_sub.clamp(min=EPS, max=1 - EPS)
        prob_del_safe = prob_del.clamp(min=EPS, max=1 - EPS)

        lp_ins = torch.where(ins_mask, torch.log(p_ins_safe), torch.log1p(-p_ins_safe)) * active_mask
        lp_del_sub = torch.where(del_sub_mask, torch.log(p_del_sub_safe), torch.log1p(-p_del_sub_safe)) * active_mask

        del_event_mask = (del_mask & del_sub_mask).float()
        sub_event_mask = (sub_mask & del_sub_mask).float()
        lp_split = torch.log(prob_del_safe) * del_event_mask + torch.log1p(-prob_del_safe) * sub_event_mask

        ins_token_probs = torch.gather(ins_probs, 2, ins_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)
        sub_token_probs = torch.gather(sub_probs, 2, sub_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(EPS)

        lp_tokens = torch.log(ins_token_probs) * (ins_mask & non_pad_mask).float()
        lp_tokens += torch.log(sub_token_probs) * (sub_mask & non_pad_mask).float()

        step_logprob = (lp_ins + lp_del_sub + lp_split + lp_tokens).sum(dim=1)

    t_next = t + adapt_h
    return x_next, t_next, step_logprob

def sample(
    guidance: Optional[Callable] = None,
    return_logprob: bool = False,
    target_y: Optional[torch.Tensor] = None,
    n_samples: int = 1,
    n_steps: int = 1000,
    initial_x: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    beta: int = 5 # reward sharpening
):
    """Shared sampler for unguided and guided inference, from x0 to x1."""
    model.eval()
    default_h = 1.0 / n_steps

    if initial_x is not None:
        x_t = initial_x.detach().cpu().clone()
    else:
        print(f"DEBUG: initial_x is None for {guidance}")
        x_0, _, _, _, _, _ = make_batch(batch_size=n_samples, min_length=min_seq_len, max_length=max_seq_len,
            vocab_size=V, coupling=coupling, seq_align_fn=seq_align_fn, num_cycles_fn=num_cycles_fn, x_int_fn=x_int_fn)
        x_t = x_0.detach().cpu()

    t = torch.zeros(x_t.shape[0], 1) # start from t=0
    step_logprobs = []
    trajectory = [x_t.clone()] if return_trajectory else None

    while t.max().item() <= (1.0 - default_h): # loops from t=0 to T, backed by CTMC
        x_t, t, step_lp = _ctmc_step(
            x_t=x_t,
            t=t,
            default_h=default_h,
            guidance=guidance,
            target_y=target_y,
            return_logprob=return_logprob,
            beta = beta
        )
        if return_trajectory:
            trajectory.append(x_t.clone())
        if return_logprob and step_lp is not None:
            step_logprobs.append(step_lp)

    result = {
        "final": x_t,
        "trajectory": trajectory,
        "logprob": sequence_logprob(step_logprobs) if return_logprob else None,
    }
    return result

# %%
# key utils and algorithms for guidance baselines
def best_of_k(
    target_y: torch.Tensor,
    initial_x: torch.Tensor,
    n_steps: int = 1000,
    alpha: int = 10,
    max_attempts: int = 20,
):
    """Best_of_k sampling, as a baseline."""
    start, best = time.perf_counter(), None
    for attempts in range(1, max_attempts + 1):
        out = sample(guidance=None, return_logprob=True, target_y=target_y, n_samples=1, 
                     n_steps=n_steps, initial_x=initial_x, return_trajectory=False)
        x_final = out["final"][0]
        r = float(edit_distance_reward(x_final, target_y, alpha=alpha))
        lp = float(out["logprob"][0].item())
        if best is None or r > best["reward"]:
            best = {"final": x_final,
                "logprob": lp,
                "reward": r,
                "attempts": attempts}
        # if np.random.rand() < r:
        #     return {"final": x_final,"logprob": lp,"reward": r,"time": float(time.perf_counter() - start),
        #               "attempts": attempts,"accepted": True}
    return {"final": best["final"], # type: ignore
        "logprob": best["logprob"], # type: ignore
        "reward": best["reward"], # type: ignore
        "time": float(time.perf_counter() - start),
        "attempts": max_attempts,
        "accepted": False}

def bootstrap_smc_sample(
    target_y: torch.Tensor,
    initial_x: torch.Tensor,
    n_particles: int = 8,
    n_steps: int = 1000,
    ess_threshold: Optional[float] = None,
    alpha: int = 5,
    beta: int = 5
):
    """Bootstrap SMC with q=p, starting all particles from the same paired initial_x."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    if ess_threshold is None:
        ess_threshold = n_particles / 2

    start = time.perf_counter()
    default_h = 1.0 / n_steps

    if initial_x.dim() == 1:
        initial_x = initial_x.unsqueeze(0)
    particles = initial_x.detach().cpu().repeat(n_particles, 1)
    t = torch.zeros(n_particles, 1)

    logw = torch.zeros(n_particles)
    traj_logprob = torch.zeros(n_particles)

    n_workers = min(n_particles, max(1, os.cpu_count() or 1))

    def _particle_reward(x_particle: torch.Tensor) -> float:
        return float(edit_distance_reward(x_particle, target_y, alpha=alpha))

    def _batch_rewards(xs: torch.Tensor, reward_power: int = 1) -> torch.Tensor:
        if n_workers <= 1:
            vals = [_particle_reward(xs[i]) for i in range(xs.shape[0])]
        else:
            vals = list(executor.map(_particle_reward, [xs[i] for i in range(xs.shape[0])]))
        rewards = torch.tensor(vals, dtype=torch.float32)
        if reward_power != 1:
            rewards = rewards.pow(reward_power)
        return rewards

    executor = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        curr_reward = _batch_rewards(particles, reward_power=1)  # keep raw reward (r^1) throughout

        while t.max().item() <= (1.0 - default_h):
            particles, t, step_lp = _ctmc_step(
                x_t=particles,
                t=t,
                default_h=default_h,
                guidance=None,
                target_y=target_y,
                return_logprob=True,
            )
            if step_lp is None:
                step_lp = torch.zeros(n_particles)
            traj_logprob += step_lp

            next_reward = _batch_rewards(particles, reward_power=1)  # raw r_next

            # Weight ratio: r(x_next)^beta / r(x_curr)^beta — apply beta only here
            logw += torch.log(next_reward.pow(beta) + EPS) - torch.log(curr_reward.pow(beta) + EPS)
            curr_reward = next_reward  # always raw reward

            max_logw = torch.max(logw)
            w = torch.exp(logw - max_logw)
            w = w / w.sum()

            if effective_sample_size(w) < ess_threshold:
                idx = torch.multinomial(w, num_samples=n_particles, replacement=True)
                particles = particles[idx]
                traj_logprob = traj_logprob[idx]
                curr_reward = curr_reward[idx]
                logw = torch.zeros(n_particles)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    max_logw = torch.max(logw)
    final_w = torch.exp(logw - max_logw)
    final_w = final_w / final_w.sum()
    chosen_idx = int(torch.multinomial(final_w, num_samples=1).item())

    return {
        "final": particles[chosen_idx].clone(),
        "logprob": float(traj_logprob[chosen_idx].item()),
        "reward": float(curr_reward[chosen_idx].item()),  # raw reward (r^1)
        "time": float(time.perf_counter() - start),
    }

def exact_guidance_u(
    x_t: torch.Tensor,
    lambda_ins: torch.Tensor,
    lambda_sub: torch.Tensor,
    lambda_del: torch.Tensor,
    ins_probs: torch.Tensor,
    sub_probs: torch.Tensor,
    target_y: torch.Tensor,
    alpha: int = 5, # reward function parameter
    beta: int = 5 # reward sharpening
    ):
    """
    Token-level exact guidance on CTMC rates u, then re-parameterize to (lambda, Q).
    """
    u_ins = lambda_ins.unsqueeze(-1) * ins_probs
    u_sub = lambda_sub.unsqueeze(-1) * sub_probs
    u_del = lambda_del.clone()

    # Guide CTMC rates directly, then map back to (lambda, Q).
    u_ins_guided = u_ins.clone()
    u_sub_guided = u_sub.clone()
    u_del_guided = u_del.clone()

    batch_size, seq_len = x_t.shape
    vocab_size = ins_probs.shape[-1]

    # Optional sanity check for CTMC diagonal convention on active (non-pad) sites.
    # For each state component, q(x,x) should be -sum_{x'!=x} q(x'|x).
    if False:
        active_mask = (x_t != PAD_TOKEN)
        total_out = (
            u_ins.sum(dim=-1)
            + u_sub.sum(dim=-1)
            + u_del
        )
        q_xx = -total_out
        resid = (q_xx + total_out)[active_mask]
        if resid.numel() > 0:
            print("CTMC residual mean:", float(resid.abs().mean().item()))
            print("CTMC residual max:", float(resid.abs().max().item()))

    for b in range(batch_size):
        x_tokens = trim_pad(x_t[b])
        base_reward = max(edit_distance_reward(x_tokens, target_y, alpha=alpha), EPS)
        cache: Dict[tuple, float] = {}

        def cached_reward(tokens):
            key = tuple(tokens)
            if key not in cache:
                cache[key] = edit_distance_reward(tokens, target_y, alpha=alpha)
            return cache[key]

        for i in range(seq_len):
            if int(x_t[b, i].item()) == PAD_TOKEN:
                u_ins_guided[b, i] = 0.0
                u_sub_guided[b, i] = 0.0
                u_del_guided[b, i] = 0.0
                continue

            ins_rates = u_ins[b, i].clone()
            sub_rates = u_sub[b, i].clone()

            for tok in range(vocab_size):
                x_ins = _virtual_apply_edit(x_tokens, "ins", i, tok)
                x_sub = _virtual_apply_edit(x_tokens, "sub", i, tok)
                ins_rates[tok] *= (float(cached_reward(x_ins) / base_reward)) ** beta
                sub_rates[tok] *= (float(cached_reward(x_sub) / base_reward)) ** beta

            x_del = _virtual_apply_edit(x_tokens, "del", i)
            del_ratio = (float(cached_reward(x_del) / base_reward)) ** beta

            u_ins_guided[b, i] = torch.clamp(ins_rates, min=0.0)
            u_sub_guided[b, i] = torch.clamp(sub_rates, min=0.0)
            u_del_guided[b, i] = torch.clamp(u_del[b, i] * max(del_ratio, 0.0), min=0.0)

    lambda_ins_guided = u_ins_guided.sum(dim=-1)
    lambda_sub_guided = u_sub_guided.sum(dim=-1)
    lambda_del_guided = u_del_guided

    # Default fallback keeps Q valid even when guided total rates collapse to zero.
    ins_probs_fallback = ins_probs / ins_probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    sub_probs_fallback = sub_probs / sub_probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
    ins_probs_guided = ins_probs_fallback.clone()
    sub_probs_guided = sub_probs_fallback.clone()

    ins_positive = lambda_ins_guided > EPS
    sub_positive = lambda_sub_guided > EPS

    if ins_positive.any():
        ins_probs_guided[ins_positive] = (
            u_ins_guided[ins_positive] / (lambda_ins_guided[ins_positive].unsqueeze(-1) + EPS)
        )
        ins_row_sum = ins_probs_guided[ins_positive].sum(dim=-1, keepdim=True).clamp_min(EPS)
        ins_probs_guided[ins_positive] = torch.clamp(ins_probs_guided[ins_positive] / ins_row_sum, min=0.0)

    if sub_positive.any():
        sub_probs_guided[sub_positive] = (
            u_sub_guided[sub_positive] / (lambda_sub_guided[sub_positive].unsqueeze(-1) + EPS)
        )
        sub_row_sum = sub_probs_guided[sub_positive].sum(dim=-1, keepdim=True).clamp_min(EPS)
        sub_probs_guided[sub_positive] = torch.clamp(sub_probs_guided[sub_positive] / sub_row_sum, min=0.0)

    return (
        lambda_ins_guided,
        lambda_sub_guided,
        lambda_del_guided,
        ins_probs_guided,
        sub_probs_guided,
    )

# %%
# Extended evaluation helpers (single-pass metrics + optional trajectory capture)
from typing import Any

METHOD_SPECS = [
    ("unguided", "Unguided"),
    ("best_of_k", "Rejection Sampling"),
    ("bootstrap_smc", "Bootstrap SMC"),
    ("exact_guidance_u", "Exact Guidance"),
]
METHOD_KEY_TO_LABEL = {k: v for k, v in METHOD_SPECS}
METHOD_LABEL_TO_KEY = {v: k for k, v in METHOD_SPECS}


def run_single_trial(
    method_key: str,
    target_pair_y: torch.Tensor,
    initial_pair_x: torch.Tensor,
    n_steps: int,
    n_particles: int,
    alpha: int,
    max_rejection_attempts: int,
    beta: int,
    return_trajectory: bool = False,
):
    if method_key == "unguided":
        t0 = time.perf_counter()
        out = sample(
            guidance=None,
            return_logprob=True,
            target_y=target_pair_y,
            n_samples=1,
            n_steps=n_steps,
            initial_x=initial_pair_x,
            return_trajectory=return_trajectory,
        )
        elapsed = time.perf_counter() - t0
        x_final = out["final"][0]
        lp = float(out["logprob"][0].item())
        r = float(edit_distance_reward(x_final, target_pair_y, alpha=alpha))
        trajectory = out["trajectory"] if return_trajectory else None

    elif method_key == "best_of_k":
        out = best_of_k(
            target_y=target_pair_y,
            initial_x=initial_pair_x,
            n_steps=n_steps,
            alpha=alpha,
            max_attempts=max_rejection_attempts,
        )
        elapsed = float(out["time"])
        x_final = out["final"]
        lp = float(out["logprob"])
        r = float(out["reward"])
        trajectory = None

    elif method_key == "bootstrap_smc":
        out = bootstrap_smc_sample(
            target_y=target_pair_y,
            initial_x=initial_pair_x,
            n_particles=n_particles,
            n_steps=n_steps,
            ess_threshold=n_particles / 2,
            alpha=alpha,
            beta=beta,
        )
        elapsed = float(out["time"])
        x_final = out["final"]
        lp = float(out["logprob"])
        r = float(out["reward"])
        trajectory = None

    elif method_key == "exact_guidance_u":
        t0 = time.perf_counter()
        out = sample(
            guidance=exact_guidance_u,
            return_logprob=True,
            target_y=target_pair_y,
            n_samples=1,
            n_steps=n_steps,
            initial_x=initial_pair_x,
            return_trajectory=return_trajectory,
            beta=beta,
        )
        elapsed = time.perf_counter() - t0
        x_final = out["final"][0]
        lp = float(out["logprob"][0].item())
        r = float(edit_distance_reward(x_final, target_pair_y, alpha=alpha))
        trajectory = out["trajectory"] if return_trajectory else None

    else:
        raise ValueError(f"Unknown method_key: {method_key}")

    d, d_norm = edit_distance_metrics(x_final, target_pair_y)
    return {
        "x_final": x_final,
        "reward": r,
        "logprob": lp,
        "runtime": float(elapsed),
        "edit_distance": float(d),
        "normalized_edit_distance": float(d_norm),
        "trajectory": trajectory,
    }


def benchmark_methods(
    N_eval: int = 30,
    n_steps: int = 1000,
    n_particles: int = 8,
    alpha: int = 10,
    max_rejection_attempts: int = 20,
    show_progress: bool = True,
    beta: int = 5,
    target_y: torch.Tensor = target_y,
    target_name: str = "target_y",
    eval_pairs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    method_keys: Optional[list[str]] = None,
    return_trials: bool = False,
    capture_trajectory: Optional[dict[str, Any]] = None,
):
    if method_keys is None:
        method_keys = [k for k, _ in METHOD_SPECS]

    if eval_pairs is None:
        eval_pairs = [(paired_x0, target_y) for _ in range(N_eval)]

    rows = []
    trial_rows = []
    captured_artifact = None

    for method_key in method_keys:
        method_label = METHOD_KEY_TO_LABEL[method_key]
        times, rewards, logprobs, edit_ds, norm_edit_ds = [], [], [], [], []
        iterator = tqdm(eval_pairs, desc=f"Benchmark: {method_label}", leave=True) if show_progress else eval_pairs

        for trial_idx, (x0_pair, x1_pair) in enumerate(iterator):
            target_pair_y = x1_pair
            initial_pair_x = x0_pair.unsqueeze(0)

            should_capture = bool(
                capture_trajectory is not None
                and capture_trajectory.get("method_key") == method_key
                and capture_trajectory.get("trial_idx", 0) == trial_idx
            )

            trial = run_single_trial(
                method_key=method_key,
                target_pair_y=target_pair_y,
                initial_pair_x=initial_pair_x,
                n_steps=n_steps,
                n_particles=n_particles,
                alpha=alpha,
                max_rejection_attempts=max_rejection_attempts,
                beta=beta,
                return_trajectory=should_capture,
            )

            times.append(trial["runtime"])
            rewards.append(trial["reward"])
            logprobs.append(trial["logprob"])
            edit_ds.append(trial["edit_distance"])
            norm_edit_ds.append(trial["normalized_edit_distance"])

            trial_rows.append(
                {
                    "trial": int(trial_idx),
                    "method_key": method_key,
                    "method": method_label,
                    "beta": int(beta),
                    "target": target_name,
                    "runtime": float(trial["runtime"]),
                    "reward": float(trial["reward"]),
                    "logprob": float(trial["logprob"]),
                    "edit_distance": float(trial["edit_distance"]),
                    "normalized_edit_distance": float(trial["normalized_edit_distance"]),
                }
            )

            if should_capture:
                captured_artifact = {
                    "method_key": method_key,
                    "method": method_label,
                    "beta": int(beta),
                    "target": target_name,
                    "initial_x": x0_pair.detach().cpu().clone(),
                    "target_y": target_pair_y.detach().cpu().clone(),
                    "trajectory": trial["trajectory"],
                }

        rows.append(
            {
                "method_key": method_key,
                "method": method_label,
                "beta": int(beta),
                "target": target_name,
                "avg_time": float(np.mean(times)),
                "avg_reward": float(np.mean(rewards)),
                "avg_logprob": float(np.mean(logprobs)),
                "avg_edit_distance": float(np.mean(edit_ds)),
                "avg_normalized_edit_distance": float(np.mean(norm_edit_ds)),
            }
        )

    summary_df = pd.DataFrame(rows)
    if return_trials:
        return summary_df, pd.DataFrame(trial_rows), captured_artifact
    return summary_df

# %%
# Single-pass evaluation over (method, beta, target) with cached trial metrics for plotting
import matplotlib.pyplot as plt

N_eval = 20
n_steps = 280
n_particles_eval = 10
betas = [5, 10]
_, in_dis_y, _, _, _, _ = make_batch(
        batch_size=batch_size,
        min_length=min_seq_len,
        max_length=max_seq_len,
        vocab_size=V,
        coupling=coupling,
        seq_align_fn=seq_align_fn,
        num_cycles_fn=num_cycles_fn,
        x_int_fn=x_int_fn,
    )
in_dis_y = in_dis_y[0].detach().cpu()
ys = [
    ("In-Distribution", in_dis_y),
    ("Linear", generate_line_y())
]
    # ("OOD Sine", generate_sine_y()),

results_dfs = []
all_summary_dfs = []
all_trial_dfs = []
trajectory_artifact = None

for target_name, y_target in ys:
    eval_pairs = [(paired_x0, y_target) for _ in range(N_eval)]

    for b in betas:
        capture_spec = None
        if target_name == "OOD Sine" and b == 15:
            capture_spec = {"method_key": "exact_guidance_u", "trial_idx": 0}

        summary_df, trial_df, captured = benchmark_methods(
            N_eval=N_eval,
            n_steps=n_steps,
            n_particles=n_particles_eval,
            alpha=1,
            max_rejection_attempts=10,
            show_progress=True,
            beta=b,
            target_y=y_target,
            target_name=target_name,
            eval_pairs=eval_pairs,
            return_trials=True,
            capture_trajectory=capture_spec,
        )

        results_dfs.append(summary_df)
        all_summary_dfs.append(summary_df)
        all_trial_dfs.append(trial_df)

        if captured is not None:
            trajectory_artifact = captured

summary_results_df = pd.concat(all_summary_dfs, ignore_index=True)
trial_results_df = pd.concat(all_trial_dfs, ignore_index=True)
summary_results_df.to_csv('summary_280.csv', index=False)
trial_results_df.to_csv('trials_280.csv', index=False)

# %%
# Save sequences for visualization:
def _tokens_no_pad(x: torch.Tensor) -> list[int]:
    x_cpu = x.detach().cpu()
    if x_cpu.ndim > 1:
        x_cpu = x_cpu[0]
    flat = x_cpu.reshape(-1)
    return [int(tok) for tok in flat.tolist() if int(tok) != PAD_TOKEN]


def _sequence_target_label(target_name: str) -> str:
    if target_name == "In-Distribution":
        return "in_dis_y"
    return target_name


representative_rows = []
representative_rows.append(
    {
        "record_type": "x0",
        "method_key": "",
        "method": "",
        "beta": "",
        "target": "shared",
        "sequence_json": json.dumps(_tokens_no_pad(paired_x0)),
    }
)

for target_name, y_target in ys:
    target_label = _sequence_target_label(target_name)
    representative_rows.append(
        {
            "record_type": "target",
            "method_key": "",
            "method": "",
            "beta": "",
            "target": target_label,
            "sequence_json": json.dumps(_tokens_no_pad(y_target)),
        }
    )

    for b in betas:
        for method_key, method_label in METHOD_SPECS:
            representative_trial = run_single_trial(
                method_key=method_key,
                target_pair_y=y_target,
                initial_pair_x=paired_x0.unsqueeze(0),
                n_steps=n_steps,
                n_particles=n_particles_eval,
                alpha=1,
                max_rejection_attempts=10,
                beta=b,
                return_trajectory=False,
            )
            representative_rows.append(
                {
                    "record_type": "generated",
                    "method_key": method_key,
                    "method": method_label,
                    "beta": int(b),
                    "target": target_label,
                    "sequence_json": json.dumps(_tokens_no_pad(representative_trial["x_final"])),
                }
            )

sequences_df = pd.DataFrame(representative_rows)
sequences_df.to_csv("sequences_280.csv", index=False)
