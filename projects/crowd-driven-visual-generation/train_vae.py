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
    p.add_argument("--ch-mult", default="1,2,4",
                   help="channel multiplier per stage; #entries = #downsamples. "
                        "'1,2,4' → 8× downsample (64→8, a [4,8,8] latent); "
                        "'2,4' → 4× downsample (64→16, a [4,16,16] latent — 4× more "
                        "capacity, much crisper reconstructions).")
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=5000, help="max images (HF datasets)")
    p.add_argument("--image-col", default="image", help="image column (HF datasets)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="crowd-driven-visual-generation")
    p.add_argument("--sample-every", type=int, default=5,
                   help="log live reconstruction images every N epochs (wandb)")
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="max grad norm (0 = off); also logged as grad_norm")
    p.add_argument("--perceptual", type=float, default=0.0,
                   help="LPIPS perceptual-loss weight (0 = off). recon is a per-image "
                        "sum (~300); LPIPS is a per-image distance (~0.2), so this "
                        "weight bridges the scales — ~100-200 makes it a meaningful "
                        "secondary term that sharpens edges without dominating pixels.")
    p.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex"],
                   help="LPIPS backbone (vgg = LDM default, sharper; alex = faster)")
    args = p.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ch_mult = tuple(int(m) for m in args.ch_mult.split(","))
    latent_hw = args.image_size // (2 ** len(ch_mult))
    print(f"device={device}  dataset={args.dataset}  out={args.out}")
    print(f"latent: [{args.latent_channels}, {latent_hw}, {latent_hw}] "
          f"({args.image_size}→{latent_hw}, {2**len(ch_mult)}× downsample)")

    wb = None
    if args.wandb:
        try:
            import wandb as wb
            run_name = f"vae-b{args.beta:g}"
            if args.perceptual > 0:
                run_name += f"-lpips{args.perceptual:g}"
            wb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            wb = None

    dl = make_dataloader(args.dataset, image_size=args.image_size,
                         batch_size=args.batch, num_workers=args.workers,
                         limit=args.limit, image_col=args.image_col)
    vae = ConvVAE(latent_channels=args.latent_channels, base=args.base,
                  ch_mult=ch_mult, beta=args.beta).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"ConvVAE: {n_params/1e6:.2f}M params | batches/epoch={len(dl)}")
    if wb:
        wb.watch(vae, log="all", log_freq=100)   # weight & gradient histograms

    # ---- optional LPIPS perceptual loss (frozen pretrained net) -------------- #
    lpips_fn = None
    if args.perceptual > 0:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net=args.lpips_net).to(device).eval()
        for pp in lpips_fn.parameters():        # freeze — it's a fixed metric, not trained
            pp.requires_grad_(False)
        print(f"LPIPS[{args.lpips_net}] perceptual loss ON (weight={args.perceptual})")

    first_batch = None
    step = 0
    for epoch in range(args.epochs):
        vae.train()
        t0 = time.time()
        running = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        if lpips_fn is not None:
            running["lpips"] = 0.0
        seen = 0
        for i, x in enumerate(dl):
            if first_batch is None:
                first_batch = x
            x = x.to(device)
            x_rec, mu, logvar, _ = vae(x)
            parts = vae.loss(x, x_rec, mu, logvar)   # total = recon + β·KL
            loss = parts["total"]
            if lpips_fn is not None:
                # both x and x_rec are in [-1, 1] — exactly LPIPS's expected range
                lp = lpips_fn(x_rec, x).mean()
                loss = loss + args.perceptual * lp
                parts["lpips"] = lp
            parts["total"] = loss                    # log the combined objective
            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), 1e9)  # measure only
            opt.step()
            for k in running:
                running[k] += parts[k].item()
            seen += 1
            step += 1
            if wb:
                logd = {"loss/total": parts["total"].item(),
                        "loss/recon": parts["recon"].item(),
                        "loss/kl": parts["kl"].item(),
                        "grad_norm": float(grad_norm),
                        "lr": opt.param_groups[0]["lr"]}
                if lpips_fn is not None:
                    logd["loss/lpips"] = parts["lpips"].item()
                wb.log(logd, step=step)
            if args.steps and (i + 1) >= args.steps:
                break
        avg = {k: v / max(seen, 1) for k, v in running.items()}
        extra = f" lpips={avg['lpips']:.4f}" if "lpips" in avg else ""
        print(f"epoch {epoch+1:>3}/{args.epochs} | "
              f"total={avg['total']:.2f} recon={avg['recon']:.2f} kl={avg['kl']:.2f}{extra} "
              f"| {time.time()-t0:.1f}s")
        if wb:
            ep = {"epoch/total": avg["total"], "epoch/recon": avg["recon"],
                  "epoch/kl": avg["kl"], "epoch": epoch + 1}
            if "lpips" in avg:
                ep["epoch/lpips"] = avg["lpips"]
            wb.log(ep, step=step)
            if (epoch + 1) % args.sample_every == 0 and first_batch is not None:
                live = save_reconstructions(
                    vae, first_batch, os.path.join(args.out, "recon_live.png"), device)
                if live:
                    wb.log({"recon/live": wb.Image(live)}, step=step)
                vae.train()   # save_reconstructions() left it in eval()

    ckpt = os.path.join(args.out, "vae.pt")
    torch.save({"model": vae.state_dict(),
                "config": {"base": args.base, "latent_channels": args.latent_channels,
                           "ch_mult": list(ch_mult), "image_size": args.image_size,
                           "latent_hw": latent_hw, "beta": args.beta}},
               ckpt)
    print(f"saved checkpoint → {ckpt}")

    rec_path = save_reconstructions(vae, first_batch, os.path.join(args.out, "recon.png"), device)
    print(f"saved reconstructions → {rec_path}" if rec_path
          else "(torchvision not available — skipped reconstruction image)")

    if wb:
        if rec_path:
            wb.log({"reconstructions": wb.Image(rec_path)})
        wb.finish()


if __name__ == "__main__":
    main()
