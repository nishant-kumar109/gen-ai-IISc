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

## ✅ `aggregation.py` (implemented) — the core contribution
Four permutation-invariant strategies mapping a set of embeddings → one conditioning vector:
- `mean_pool` · `cluster_centroid` — NumPy-optional, **run locally** (against HashingEncoder).
- `AttentionPooling` (Set-Transformer PMA) · `DeepSets` — learnable `torch.nn` modules (**Colab**).

```python
from models.aggregation import mean_pool, cluster_centroid   # non-learnable
c = mean_pool(emb)                       # emb: [N, D] -> conditioning vector [D]
c = cluster_centroid(emb, k=4, mode="dominant")
# learnable (Colab): AttentionPooling.build(in_dim, out_dim) / DeepSets.build(in_dim, ...)
```
Smoke test: `python3 models/aggregation.py`. This completes the **input pipeline**
(simulator → encode → aggregate).

## ✅ `vae.py` (implemented)
From-scratch convolutional VAE for the latent space — reparameterization trick + ELBO (β·KL).
Default: 64×64 RGB → spatial latent `[4, 8, 8]` (8× downsample). Diffusion runs in this latent.

```python
from models.vae import ConvVAE
vae = ConvVAE(beta=1.0)
x_rec, mu, logvar, z = vae(x)            # x: [B,3,64,64]
losses = vae.loss(x, x_rec, mu, logvar)  # {'total','recon','kl'}
z = vae.encode_to_latent(x)              # for the diffusion pipeline
```
Shape self-test: `python3 models/vae.py` (verified: 4.58M params, z=[B,4,8,8]).

## Still to build
- `diffusion.py` — conditional DDPM (U-Net), noise-prediction loss, classifier-free guidance, DDIM sampler.
- `gan.py` — conditional GAN baseline.

Configs + fixed seeds live alongside each module for reproducibility.
