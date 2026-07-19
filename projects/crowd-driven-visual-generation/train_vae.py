"""Train the ConvVAE on the visual domain (Stage 1 — the latent space).

Runs locally (CPU, synthetic data — a quick smoke test) and on Colab (GPU, real
art images). Saves a checkpoint and a before/after reconstruction grid.

Examples
--------
Local smoke test (seconds):
    python3 train_vae.py --dataset synthetic --epochs 1 --steps 20 --batch 16 --device cpu

Colab (real training):
    python3 train_vae.py --dataset cifar10 --epochs 30 --batch 128 --out runs/vae
    python3 train_vae.py --dataset /path/to/art_images --image-size 64 --epochs 50
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from data.art_dataset import make_dataloader
from models.vae import ConvVAE


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_reconstructions(vae, batch, path, device):
    """Save a grid: top row = inputs, bottom row = reconstructions."""
    vae.eval()
    with torch.no_grad():
        x = batch[:8].to(device)
        mu, logvar = vae.encode(x)
        rec = vae.decode(mu)
    grid = torch.cat([x.cpu(), rec.cpu()], dim=0)          # [16, 3, H, W]
    grid = (grid.clamp(-1, 1) + 1) / 2                     # → [0, 1]
    try:
        from torchvision.utils import save_image
        save_image(grid, path, nrow=8)
        return path
    except Exception:
        return None                                        # torchvision absent (local)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic",
                   help="synthetic | cifar10 | <folder> | <hf-dataset-id>")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=0, help="cap steps/epoch (0 = full)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (β-VAE)")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device={device}  dataset={args.dataset}  out={args.out}")

    dl = make_dataloader(args.dataset, image_size=args.image_size,
                         batch_size=args.batch, num_workers=args.workers)
    vae = ConvVAE(latent_channels=args.latent_channels, base=args.base,
                  beta=args.beta).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"ConvVAE: {n_params/1e6:.2f}M params | batches/epoch={len(dl)}")

    first_batch = None
    step = 0
    for epoch in range(args.epochs):
        vae.train()
        t0 = time.time()
        running = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        seen = 0
        for i, x in enumerate(dl):
            if first_batch is None:
                first_batch = x
            x = x.to(device)
            x_rec, mu, logvar, _ = vae(x)
            losses = vae.loss(x, x_rec, mu, logvar)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            for k in running:
                running[k] += losses[k].item()
            seen += 1
            step += 1
            if args.steps and (i + 1) >= args.steps:
                break
        avg = {k: v / max(seen, 1) for k, v in running.items()}
        print(f"epoch {epoch+1:>3}/{args.epochs} | "
              f"total={avg['total']:.2f} recon={avg['recon']:.2f} kl={avg['kl']:.2f} "
              f"| {time.time()-t0:.1f}s")

    ckpt = os.path.join(args.out, "vae.pt")
    torch.save({"model": vae.state_dict(),
                "config": {"base": args.base, "latent_channels": args.latent_channels,
                           "image_size": args.image_size, "beta": args.beta}},
               ckpt)
    print(f"saved checkpoint → {ckpt}")

    rec_path = save_reconstructions(vae, first_batch, os.path.join(args.out, "recon.png"), device)
    print(f"saved reconstructions → {rec_path}" if rec_path
          else "(torchvision not available — skipped reconstruction image)")


if __name__ == "__main__":
    main()
