# models/

Model implementations (from scratch).

- `vae/` — VAE / VQ-VAE for the latent space (ELBO / codebook).
- `diffusion/` — conditional DDPM (U-Net), noise-prediction loss, classifier-free guidance, DDIM sampler.
- `aggregation/` — permutation-invariant aggregators: mean-pool · cluster-centroid mixture · Set-Transformer (PMA) · Deep-Sets.
- `gan/` — conditional GAN baseline.
- `encoders/` — text/emoji + doodle encoders → shared latent.

Configs + fixed seeds live alongside each module for reproducibility.
