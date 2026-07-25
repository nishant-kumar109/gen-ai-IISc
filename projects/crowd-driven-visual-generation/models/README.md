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

## ✅ `diffusion.py` (implemented) — the generative core
Conditional latent DDPM: a **FiLM-conditioned U-Net** (ε-prediction) + **cosine** schedule +
**classifier-free-guidance** (condition-dropout at train, guided **DDIM** sampling with scale `w`).
The U-Net depth adapts to `ch_mult` — default `(1,2,4)` fits the `16×16` latent (16→8→4).

```python
from models.diffusion import ConditionalUNet, GaussianDiffusion
unet = ConditionalUNet(latent_channels=4, cond_dim=512, ch_mult=(1,2,4))
diff = GaussianDiffusion(unet, timesteps=1000, p_uncond=0.1)
loss = diff.p_losses(z0, cond)                  # train: z0 [B,4,16,16], cond [B,512]
z    = diff.ddim_sample((B,4,16,16), cond, w=3.0, steps=50)   # sample
```
Shape self-test: `python3 models/diffusion.py` (6.91M params).

> **Training signal (per proposal §2):** the model is trained conditioned on each image's *own*
> CLIP embedding; the crowd **aggregation** (`aggregation.py`) is applied only at **inference**.
> Trained end-to-end by **`train_diffusion.py`** + **`notebooks/02_train_diffusion.ipynb`**
> (freezes the VAE, pre-encodes latents + CLIP conds, scales latents, trains with CFG).

## ✅ Stage 3 — crowd → image (implemented, inference)
**`sample_crowd.py`** + **`notebooks/03_crowd_to_image.ipynb`**: crowd → CLIP text
embeddings → `aggregate()` → guided DDIM sampling → VAE decode. Two study modes:
`themes` (does each theme render?) and `diversity` (RQ1 punchline: mean-pool → mush,
centroid → stays coherent as off-theme noise rises). Non-learnable aggregators only;
the learnable ones (DeepSets / attention) still need a short training pass.

**Modality-gap correction** (`compute_gap.py`): the diffusion model trains on CLIP
*image* embeddings but a crowd is CLIP *text* — different regions of CLIP space. We
measure `gap = mean(image embs) − mean(text embs)` once and translate text conditions
onto the image manifold (`sample_crowd.py --gap-file`). Fixes weak/OOD text conditioning.

## ✅ Learnable aggregators (trained) — completes the 4-way RQ1 study
**`train_aggregators.py`** + **`notebooks/05_train_aggregators.ipynb`**: train DeepSets + attention-PMA
self-supervised — `loss = 1 - cos(agg(crowd_embeddings), CLIP_text("a {theme} scene"))` — to extract the
dominant theme robustly. `sample_crowd.py`/`evaluate.py` load them via `--agg-ckpt` for a 4-way comparison
(`mean · centroid · deepsets · attention`).

## Still to build
- `gan.py` — conditional GAN baseline (optional).
- Rigor: more eval repeats + error bars on the RQ1 plots.

Configs + fixed seeds live alongside each module for reproducibility.
