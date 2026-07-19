"""Permutation-invariant aggregation — the core contribution.

Collapse a *set* of ``N`` response embeddings ``{e_1 … e_N}`` into a single
**conditioning vector** ``c`` that guides the diffusion model. Because crowd
input has no meaningful order, every aggregator here is permutation-invariant.

Four strategies compared in RQ1:

    "mean"      mean-pool                 c = normalize(mean_i e_i)        (naïve baseline)
    "centroid"  cluster-centroid mixture  k-means → dominant/weighted centroid
    "attention" Set-Transformer PMA       learnable pooling-by-attention   (torch)
    "deepsets"  Deep-Sets                 c = ρ(Σ φ(e_i))                  (torch)

`mean` and `centroid` are NumPy-optional (they run locally on the HashingEncoder
stub, so we can test the whole simulator → encode → aggregate path without torch).
`attention` and `deepsets` are `torch.nn` modules (Colab).

Local smoke test (mean + centroid, no torch):
    python3 models/aggregation.py
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence

# NumPy is used when available (Colab) for speed; falls back to pure Python.
try:
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

Vector = List[float]
Matrix = Sequence[Sequence[float]]      # [N, D] — list-of-lists or ndarray


# --------------------------------------------------------------------------- #
# helpers (work on list-of-lists OR ndarray rows)
# --------------------------------------------------------------------------- #
def _rows(x: Matrix) -> List[Vector]:
    return [list(map(float, row)) for row in x]


def _normalize(v: Vector) -> Vector:
    n = math.sqrt(sum(a * a for a in v)) or 1.0
    return [a / n for a in v]


def _col_mean(rows: List[Vector]) -> Vector:
    n = len(rows)
    d = len(rows[0])
    out = [0.0] * d
    for row in rows:
        for j in range(d):
            out[j] += row[j]
    return [a / n for a in out]


def _sqdist(a: Vector, b: Vector) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# non-learnable aggregators
# --------------------------------------------------------------------------- #
def mean_pool(embeddings: Matrix) -> Vector:
    """c = normalize(mean_i e_i). The naïve baseline — averages *everything*,
    so off-theme responses drag the condition toward mush."""
    if _np is not None and isinstance(embeddings, _np.ndarray):
        c = embeddings.mean(axis=0)
        return _normalize(c.tolist())
    return _normalize(_col_mean(_rows(embeddings)))


def _kmeans(rows: List[Vector], k: int, iters: int = 25, seed: int = 0):
    """Tiny Lloyd's k-means (pure Python). Returns (labels, centroids, sizes)."""
    rng = random.Random(seed)
    k = max(1, min(k, len(rows)))
    centroids = [row[:] for row in rng.sample(rows, k)]
    labels = [0] * len(rows)
    for _ in range(iters):
        # assign
        changed = False
        for i, row in enumerate(rows):
            best, bd = 0, float("inf")
            for c_idx, c in enumerate(centroids):
                d = _sqdist(row, c)
                if d < bd:
                    bd, best = d, c_idx
            if labels[i] != best:
                labels[i] = best
                changed = True
        # update
        sums = [[0.0] * len(rows[0]) for _ in range(k)]
        counts = [0] * k
        for i, row in enumerate(rows):
            counts[labels[i]] += 1
            s = sums[labels[i]]
            for j in range(len(row)):
                s[j] += row[j]
        for c_idx in range(k):
            if counts[c_idx] > 0:
                centroids[c_idx] = [a / counts[c_idx] for a in sums[c_idx]]
        if not changed:
            break
    return labels, centroids, counts


def cluster_centroid(
    embeddings: Matrix,
    k: int = 4,
    mode: str = "dominant",
    alpha: float = 2.0,
    seed: int = 0,
) -> Vector:
    """Cluster the crowd (k-means) and summarise into one vector.

    ``mode="dominant"`` → the centroid of the **largest** cluster (most robust to
    scattered off-theme noise). ``mode="weighted"`` → size-weighted mixture of
    centroids with weights ∝ size**alpha (alpha>1 emphasises the dominant theme).
    Captures multi-modality far better than a plain mean.
    """
    if _np is not None and isinstance(embeddings, _np.ndarray):
        rows = embeddings.tolist()
    else:
        rows = _rows(embeddings)
    labels, centroids, sizes = _kmeans(rows, k, seed=seed)

    if mode == "dominant":
        dom = max(range(len(sizes)), key=lambda i: sizes[i])
        return _normalize(centroids[dom])

    # weighted mixture
    weights = [s ** alpha for s in sizes]
    wsum = sum(weights) or 1.0
    d = len(rows[0])
    mix = [0.0] * d
    for c, w in zip(centroids, weights):
        for j in range(d):
            mix[j] += (w / wsum) * c[j]
    return _normalize(mix)


# non-learnable dispatch
def aggregate(name: str, embeddings: Matrix, **kw) -> Vector:
    if name == "mean":
        return mean_pool(embeddings)
    if name == "centroid":
        return cluster_centroid(embeddings, **kw)
    raise ValueError(f"{name!r} is learnable (torch) — use its nn.Module directly")


# --------------------------------------------------------------------------- #
# learnable aggregators (torch — Colab)
# --------------------------------------------------------------------------- #
def _torch():
    import torch  # local import so the module loads without torch
    import torch.nn as nn
    return torch, nn


class DeepSets:  # thin factory returning an nn.Module (built lazily)
    """Deep-Sets: c = ρ(Σ_i φ(e_i)) — permutation-invariant by the sum.

    Usage (Colab):
        agg = DeepSets.build(in_dim=512, hidden=256, out_dim=512)
        c = agg(x)          # x: [B, N, D] → [B, out_dim]
    """

    @staticmethod
    def build(in_dim: int, hidden: int = 256, out_dim: int = 512):
        torch, nn = _torch()

        class _DeepSets(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden), nn.GELU(),
                )
                self.rho = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, out_dim),
                )

            def forward(self, x, mask=None):
                h = self.phi(x)                       # [B, N, H]
                if mask is not None:                  # mask: [B, N] (1=valid)
                    h = h * mask.unsqueeze(-1)
                    pooled = h.sum(1)                 # sum over set
                else:
                    pooled = h.sum(1)
                c = self.rho(pooled)                  # [B, out_dim]
                return c / c.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        return _DeepSets()


class AttentionPooling:  # Set-Transformer PMA (pooling by multihead attention)
    """PMA: a learnable seed query attends over the set → one pooled vector.

    Usage (Colab):
        agg = AttentionPooling.build(in_dim=512, out_dim=512, heads=4)
        c = agg(x)          # x: [B, N, D] → [B, out_dim]
    """

    @staticmethod
    def build(in_dim: int, out_dim: int = 512, heads: int = 4):
        torch, nn = _torch()

        class _PMA(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(in_dim, out_dim)
                self.seed = nn.Parameter(torch.randn(1, 1, out_dim))
                self.attn = nn.MultiheadAttention(out_dim, heads, batch_first=True)
                self.ff = nn.Sequential(
                    nn.Linear(out_dim, out_dim), nn.GELU(),
                    nn.Linear(out_dim, out_dim),
                )
                self.norm = nn.LayerNorm(out_dim)

            def forward(self, x, mask=None):
                B = x.shape[0]
                kv = self.proj(x)                         # [B, N, out]
                q = self.seed.expand(B, -1, -1)           # [B, 1, out]
                key_padding = None if mask is None else (mask == 0)  # True=pad
                pooled, _ = self.attn(q, kv, kv, key_padding_mask=key_padding)
                pooled = pooled.squeeze(1)                # [B, out]
                c = self.norm(pooled + self.ff(pooled))
                return c / c.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        return _PMA()


AGGREGATORS = {
    "mean": "non-learnable · mean_pool()",
    "centroid": "non-learnable · cluster_centroid()",
    "attention": "learnable (torch) · AttentionPooling.build()",
    "deepsets": "learnable (torch) · DeepSets.build()",
}


# --------------------------------------------------------------------------- #
# local smoke test (mean + centroid, no torch)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import os
    import sys

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    sys.path.insert(0, root)
    from data.crowd_simulator import CrowdSimulator          # noqa: E402
    from models.encoders import HashingEncoder               # noqa: E402

    enc = HashingEncoder(dim=64, seed=0)
    sim = CrowdSimulator(seed=3)

    print("aggregators:", list(AGGREGATORS), "\n")
    for div in (0.1, 0.6):
        crowd = sim.sample("paradise", n=500, diversity=div)
        emb = enc.encode_crowd(crowd)
        cm = mean_pool(emb)
        cc = cluster_centroid(emb, k=4, mode="dominant")
        # cosine similarity between the two conditioning vectors
        cos = sum(a * b for a, b in zip(cm, cc))
        print(f"diversity={div}")
        print(f"  mean_pool      : dim={len(cm)}  |c|={math.sqrt(sum(a*a for a in cm)):.3f}")
        print(f"  cluster_centroid: dim={len(cc)}  |c|={math.sqrt(sum(a*a for a in cc)):.3f}")
        print(f"  cos(mean, centroid) = {cos:.3f}\n")

    print("[note] HashingEncoder is non-semantic, so this only verifies shapes/"
          "plumbing/determinism — semantic behaviour is validated on Colab with CLIP.")


if __name__ == "__main__":
    _demo()
