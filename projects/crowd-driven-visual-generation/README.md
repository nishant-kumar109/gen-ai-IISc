# Crowd-Driven Visual Generation with Conditional Diffusion Models

Course project for **Generative AI – Principles and Applications** (IISc CCE), by Nishant Kumar.

Generate a single **coherent artwork** by conditioning a diffusion model on a **crowd** of thousands of heterogeneous inputs (words / emojis / doodles) — a *many-to-one* conditioning problem. The core study compares **permutation-invariant aggregation strategies** for diffusion conditioning, benchmarked against a GAN.

- 📄 **Proposal (formal template):** [PROPOSAL.md](PROPOSAL.md) · PDF: `project-proposal-nishant.pdf`
- 📘 **Extended summary / vision:** [SUMMARY.md](SUMMARY.md)

## Repository layout
```
crowd-driven-visual-generation/
├── PROPOSAL.md                     # project proposal (formal template)
├── SUMMARY.md                      # extended summary / vision
├── project-proposal-nishant.pdf    # proposal (PDF)
├── data/    # crowd-response simulator + dataset prep (QuickDraw doodles, word/emoji embeddings, art images)
├── models/  # VAE/VQ-VAE, conditional DDPM (U-Net) + CFG, aggregation modules, GAN baseline
└── eval/    # metrics (FID, CLIP fidelity, coherence, PPL, latency), ablation scripts
```

## Core components (planned)
- **Encoders** → shared latent for words/emojis/doodles
- **Aggregation** → mean-pool · cluster-centroid · Set-Transformer (PMA) · Deep-Sets
- **Conditional DDPM** in VAE/VQ-VAE latent space + classifier-free guidance
- **Sampling** → DDPM vs DDIM
- **Animation** → slerp between per-"song" conditioning vectors
- **Baseline** → conditional GAN

## Status
Proposal stage. See [PROPOSAL.md](PROPOSAL.md) for problem, approach, related work, evaluation plan, and timeline.
