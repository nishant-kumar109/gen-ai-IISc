# Crowd-Driven Visual Generation with Conditional Diffusion Models

Course project for **Generative AI – Principles and Applications** (IISc CCE), by Nishant Kumar.

Turn a **crowd** of hundreds of heterogeneous responses (words, emojis, doodle labels) into one
**coherent image** — a *many-to-one* conditioning problem. The study compares
**permutation-invariant aggregation strategies** for conditioning a from-scratch latent-diffusion
model, with a conditional GAN as a second generative backbone.

- 📄 **Proposal:** [PROPOSAL.md](PROPOSAL.md) · `project-proposal-nishant.pdf`
- 📘 **Extended summary:** [SUMMARY.md](SUMMARY.md)
- 📕 **Final report:** [REPORT.md](REPORT.md) · `project-report-nishant.pdf`

## Key result (RQ1)
**Learned** permutation-invariant aggregation (attention-PMA ≈ Deep-Sets) **significantly
outperforms heuristic pooling** (mean ≈ centroid) at producing on-theme images — a 0.034 CLIP
theme-fidelity gap (95% CI [0.031, 0.036], *p* ≈ 3×10⁻¹⁰³) that holds as crowd diversity rises and
**reproduces on both the diffusion and GAN backbones**. Reaching it required diagnosing and
mitigating the **CLIP modality gap** (image-trained conditioning responds weakly to text) via a
text-conditioned retrain.

## Pipeline
`crowd → CLIP embeddings → aggregate (mean / centroid / Deep-Sets / attention-PMA) → conditioning
vector → conditional generator (diffusion or GAN) in the VAE latent space → decode → image.`
The VAE is trained once and frozen; aggregation is applied only at inference, so the aggregator is a
swappable study variable.

## Repository layout
```
crowd-driven-visual-generation/
├── data/
│   ├── crowd_simulator.py       # themed crowd generator (words/emojis/doodle labels; diversity knob)
│   └── art_dataset.py           # WikiArt streaming loader (+ genre labels)
├── models/
│   ├── vae.py                   # convolutional β-VAE (latent space) + LPIPS
│   ├── diffusion.py             # FiLM-conditioned U-Net DDPM + classifier-free guidance + DDIM
│   ├── gan.py                   # conditional GAN baseline (projection discriminator, hinge loss)
│   ├── aggregation.py           # mean · cluster-centroid · Deep-Sets · attention-PMA
│   └── encoders.py              # CLIP encoder (+ hashing stub for local tests)
├── train_vae.py                 # Stage 1 — VAE
├── train_diffusion.py           # Stage 2 — conditional latent diffusion (+ --text-cond retrain)
├── train_gan.py                 # GAN baseline
├── train_aggregators.py         # learnable aggregators (Deep-Sets / attention-PMA)
├── compute_gap.py               # CLIP modality-gap vector
├── sample_crowd.py              # crowd → image inference (--gan-ckpt / --agg-ckpt / --gap-file)
├── evaluate.py                  # RQ1 metrics: CLIP theme-fidelity + consistency (mean ± SE)
├── notebooks/                   # 01–07 Colab notebooks
├── images-and-plots/            # result figures used in the report
├── REPORT.md · project-report-nishant.pdf
└── requirements.txt
```

## Components (implemented)
- **Encoders** — CLIP (ViT-B/32) shared image–text embedding space.
- **Aggregation** — mean-pool · cluster-centroid · Deep-Sets · attention-PMA (the RQ1 study variable).
- **VAE** — convolutional β-VAE, `[4,16,16]` latent, LPIPS perceptual loss.
- **Diffusion** — FiLM-conditioned U-Net (ε-prediction), cosine schedule, classifier-free guidance,
  DDIM sampling. Text-conditioned retrain bridges the CLIP modality gap.
- **GAN baseline** — conditional GAN in the same latent, for a diffusion-vs-GAN comparison.
- **Evaluation** — CLIP theme-fidelity + output consistency, per aggregator × crowd diversity, with
  error bars and a Welch's t-test. *(FID/IS are deliberately not used — there is no paired ground
  truth for a crowd.)*

## Notebooks (Colab, run in order)
1. **01_train_vae** — VAE latent space
2. **02_train_diffusion** — conditional latent diffusion
3. **03_crowd_to_image** — crowd → image + first RQ1 numbers
4. **04_retrain_diffusion** — text-conditioned retrain (fixes the modality gap)
5. **05_train_aggregators** — learnable aggregators + 4-way RQ1
6. **06_qualitative_v2** — qualitative crowd → image grids
7. **07_train_gan** — GAN baseline + diffusion-vs-GAN comparison

Each notebook auto-locates checkpoints from Google Drive or the HuggingFace Hub; training logs to
Weights & Biases. VAE/diffusion run on a T4; the retrain and GAN use an A100.

## Status
**Complete.** See [REPORT.md](REPORT.md) for the full method, results (RQ1 + diffusion-vs-GAN),
honest limitations (incl. the CLIP objective/metric alignment), and reproducibility notes.
