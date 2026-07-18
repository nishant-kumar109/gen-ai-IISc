# eval/

Evaluation and ablation scripts. Each research question maps to metrics + baselines:

| RQ | Metric(s) | Baseline |
|----|-----------|----------|
| RQ1 aggregation → coherence | coherence, CLIP theme-fidelity, FID, diversity | mean-pool |
| RQ2 diffusion vs GAN | FID, coherence, CLIP fidelity | conditional GAN |
| RQ3 interpolation animation | perceptual path length (PPL) | linear vs slerp |
| RQ4 DDPM vs DDIM | FID vs #steps, latency | DDPM (full steps) |

**Ablations:** aggregation strategy · number of inputs N · guidance scale w · clustering k · conditioning mechanism (cross-attention vs FiLM).
