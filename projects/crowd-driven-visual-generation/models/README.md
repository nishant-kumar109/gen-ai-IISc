# models/

Model implementations (from scratch).

## ✅ `encoders.py` (implemented)
Maps audience responses → a **shared embedding space**. Common `Encoder` interface with two backends:
- `CLIPEncoder` — real encoder (open_clip); text & images share one space. **Colab / torch.**
- `HashingEncoder` — deterministic, dependency-free stub for local plumbing tests (not semantic).

```python
from models.encoders import HashingEncoder     # or CLIPEncoder on Colab
emb = HashingEncoder(dim=64).encode_crowd(crowd)   # -> [N, D] unit vectors
```
Smoke test: `python3 models/encoders.py`.

## Still to build
- `aggregation.py` — permutation-invariant aggregators: mean-pool · cluster-centroid · Set-Transformer (PMA) · Deep-Sets.
- `vae.py` — VAE / VQ-VAE for the latent space (ELBO / codebook).
- `diffusion.py` — conditional DDPM (U-Net), noise-prediction loss, classifier-free guidance, DDIM sampler.
- `gan.py` — conditional GAN baseline.

Configs + fixed seeds live alongside each module for reproducibility.
