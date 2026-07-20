"""Encoders: map audience responses into a shared embedding space.

Words, emojis and doodle *labels* are all short strings, so they go through a
text encoder; actual doodle *sketches* (if we later fetch QuickDraw images) go
through an image encoder. We use CLIP so text and images live in **one** space —
which is what lets the aggregation module combine heterogeneous inputs.

Two implementations behind a common `Encoder` interface:

* ``CLIPEncoder``    — the real encoder (open_clip). Runs on Colab / any torch box.
                       Returns L2-normalised ``np.float32`` arrays.
* ``HashingEncoder`` — a deterministic, dependency-free stub. Not semantic, but it
                       lets us develop and smoke-test the *downstream* pipeline
                       (aggregation → conditioning) locally, then swap in CLIP.

Both return, for a crowd of ``N`` responses, an ``[N, D]`` array of unit vectors.

Run a local smoke test (uses HashingEncoder, no torch needed):
    python3 models/encoders.py
"""
from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:                       # avoid a hard import at runtime
    from data.crowd_simulator import Crowd


class Encoder:
    """Common interface. ``dim`` is the embedding dimensionality."""
    dim: int

    def encode_texts(self, texts: Sequence[str]):
        raise NotImplementedError

    def encode_images(self, images):
        raise NotImplementedError

    def encode_crowd(self, crowd: "Crowd"):
        """Encode every response in a crowd → ``[N, D]``.

        All current modalities (word / emoji / doodle-label) are strings, so
        they share the text path. (Duck-typed: only needs ``crowd.responses``
        with ``.value``.)
        """
        texts = [r.value for r in crowd.responses]
        return self.encode_texts(texts)


class HashingEncoder(Encoder):
    """Deterministic hash-based pseudo-embeddings — pure standard library.

    Each string is mapped to a fixed-dim unit vector via repeated SHA-256. This
    is **not** semantically meaningful (unrelated words are near-orthogonal), so
    it's only for testing plumbing/shapes and reproducibility locally. Replace
    with ``CLIPEncoder`` for real experiments.
    """

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def _embed_one(self, text: str) -> List[float]:
        vals: List[float] = []
        h = hashlib.sha256(f"{self.seed}:{text}".encode("utf-8"))
        while len(vals) < self.dim:
            h = hashlib.sha256(h.digest())
            for b in h.digest():
                if len(vals) >= self.dim:
                    break
                vals.append((b / 255.0) * 2.0 - 1.0)      # in [-1, 1]
        norm = math.sqrt(sum(x * x for x in vals)) or 1.0
        return [x / norm for x in vals]

    def encode_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def encode_images(self, images):
        raise NotImplementedError("HashingEncoder handles text/labels only")


class CLIPEncoder(Encoder):
    """Real encoder backed by CLIP (open_clip). Text and images share one space.

    Requires ``torch`` + ``open_clip`` (installed via requirements.txt on Colab).
    ``encode_*`` return L2-normalised ``np.ndarray`` of shape ``[N, D]``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        import open_clip
        import torch

        self._torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # probe the embedding dimension once
        self.dim = int(self.encode_texts(["_probe_"]).shape[1])

    def encode_texts(self, texts: Sequence[str]):
        import numpy as np  # noqa: F401  (kept for return-type clarity)

        tokens = self.tokenizer(list(texts)).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def encode_images(self, images):
        batch = self._torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()


# --------------------------------------------------------------------------- #
# Local smoke test (HashingEncoder — no torch needed)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from data.crowd_simulator import CrowdSimulator  # noqa: E402

    enc = HashingEncoder(dim=64, seed=0)
    sim = CrowdSimulator(seed=1)
    crowd = sim.sample("paradise", n=8, diversity=0.2)

    emb = enc.encode_crowd(crowd)
    n, d = len(emb), len(emb[0])
    print(f"encoder     : HashingEncoder(dim={enc.dim})")
    print(f"crowd        : theme={crowd.theme!r}, N={crowd.n}")
    print(f"embeddings   : {n} x {d}")

    # unit-norm check
    norms = [math.sqrt(sum(x * x for x in v)) for v in emb]
    print(f"norms ~1.0   : min={min(norms):.4f} max={max(norms):.4f}")

    # determinism check (same string -> same vector)
    a = enc.encode_texts(["ocean"])[0]
    b = enc.encode_texts(["ocean"])[0]
    print(f"deterministic: {a == b}")

    # identical responses share an embedding; different ones differ
    dup = enc.encode_texts(["ocean", "ocean", "fire"])
    print(f"same==same   : {dup[0] == dup[1]} | same!=diff: {dup[0] != dup[2]}")


if __name__ == "__main__":
    _demo()
