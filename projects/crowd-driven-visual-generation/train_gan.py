"""Train the conditional GAN baseline (Stage-2 alternative to diffusion).

Same latent space + CLIP conditioning as the diffusion model, so it's a fair
comparison and shares the inference pipeline. Reuses the diffusion precompute
(latents + image/text conditions). Hinge GAN loss, projection discriminator.

Local plumbing test:
    python3 train_gan.py --vae /tmp/vs/vae.pt --dataset synthetic --fake-clip \
        --epochs 1 --steps 5 --batch 8 --device cpu --out /tmp/gan

Colab (real, fair vs diffusion_v2):
    python3 train_gan.py --vae runs/vae/vae.pt --dataset huggan/wikiart --limit 15000 \
        --text-cond --p-text 0.5 --epochs 120 --base 128 --out runs/gan --wandb
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from models.gan import (Discriminator, Generator, d_hinge_loss, g_hinge_loss)
from train_diffusion import (clip_embed, load_vae, make_dataloader, precompute,
                             precompute_text)


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def sample_grid(G, vae, conds, preview, scale, device, path):
    """Grid: top = real paintings, bottom = GAN generations from the same conds."""
    G.eval()
    lat = G.sample(conds.to(device))                 # [k, C, hw, hw] (scaled space)
    imgs = vae.decode(lat / scale).clamp(-1, 1)
    grid = torch.cat([preview.to(device), imgs], dim=0)
    grid = (grid.cpu() + 1) / 2
    try:
        from torchvision.utils import save_image
        save_image(grid, path, nrow=conds.shape[0])
        return path
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="runs/vae/vae.pt")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--steps", type=int, default=0, help="cap steps/epoch (0 = full)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--base", type=int, default=128)
    p.add_argument("--d-steps", type=int, default=1, help="D updates per G update")
    p.add_argument("--text-cond", action="store_true")
    p.add_argument("--p-text", type=float, default=0.5)
    p.add_argument("--label-col", default="genre")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--fake-clip", action="store_true")
    p.add_argument("--clip-grad", type=float, default=0.0)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--out", default="runs/gan")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--image-col", default="image")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="crowd-driven-visual-generation")
    args = p.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device={device}  vae={args.vae}  dataset={args.dataset}  out={args.out}")

    vae, vcfg = load_vae(args.vae, device)
    lc = vcfg["latent_channels"]
    hw = vcfg.get("latent_hw") or (vcfg["image_size"] // (2 ** len(vcfg["ch_mult"])))
    print(f"loaded VAE: latent [{lc}, {hw}, {hw}]  (frozen)")

    clip_model, tokenizer = None, None
    if not args.fake_clip:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrained)
        clip_model = clip_model.to(device).eval()
        for pp in clip_model.parameters():
            pp.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(args.clip_model)

    wb = None
    if args.wandb:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name="gan", config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            wb = None

    # ---- precompute latents + conditions (same as diffusion) ----
    print("pre-encoding latents + conditions...")
    image_conds = text_conds = conds = None
    if args.text_cond:
        if args.fake_clip:
            N = min(args.limit, 256)
            latents = torch.randn(N, lc, hw, hw)
            image_conds = torch.nn.functional.normalize(torch.randn(N, 512), dim=-1)
            text_conds = torch.nn.functional.normalize(torch.randn(N, 512), dim=-1)
            preview = torch.rand(8, 3, args.image_size, args.image_size) * 2 - 1
        else:
            latents, image_conds, text_conds, preview = precompute_text(
                vae, clip_model, tokenizer, args.dataset, args.image_size, args.limit,
                args.image_col, args.label_col, args.batch, device)
        cond_dim = image_conds.shape[1]
    else:
        dl = make_dataloader(args.dataset, image_size=args.image_size, batch_size=args.batch,
                             shuffle=False, num_workers=args.workers,
                             limit=args.limit, image_col=args.image_col)
        latents, conds, preview = precompute(vae, clip_model, dl, device, args.fake_clip)
        cond_dim = conds.shape[1]

    scale = 1.0 / (latents.std().item() + 1e-8)
    latents = latents * scale
    print(f"latent scale={scale:.4f}  cond_dim={cond_dim}"
          + (f"  | text-cond mix p={args.p_text}" if args.text_cond else ""))

    ds = (torch.utils.data.TensorDataset(latents, image_conds, text_conds)
          if args.text_cond else torch.utils.data.TensorDataset(latents, conds))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    G = Generator(z_dim=args.z_dim, cond_dim=cond_dim, latent_channels=lc,
                  hw=hw, base=args.base).to(device)
    D = Discriminator(cond_dim=cond_dim, latent_channels=lc, hw=hw, base=args.base).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    print(f"Generator {sum(p.numel() for p in G.parameters())/1e6:.2f}M | "
          f"Discriminator {sum(p.numel() for p in D.parameters())/1e6:.2f}M | "
          f"batches/epoch={len(loader)}")
    if wb:
        wb.watch(G, log="all", log_freq=200)

    n_prev = preview.shape[0]
    fixed_conds = (text_conds if args.text_cond else conds)[:n_prev].clone()

    def get_cond(batch):
        if args.text_cond:
            z0, ic, tc = (b.to(device) for b in batch)
            use_text = (torch.rand(z0.shape[0], device=device) < args.p_text)[:, None]
            return z0, torch.where(use_text, tc, ic)
        return batch[0].to(device), batch[1].to(device)

    step = 0
    for epoch in range(args.epochs):
        G.train(); D.train()
        t0, rd, rg, seen = time.time(), 0.0, 0.0, 0
        for i, batch in enumerate(loader):
            z0, cond = get_cond(batch)
            B = z0.shape[0]
            # ---- D ----
            for _ in range(args.d_steps):
                zt = torch.randn(B, args.z_dim, device=device)
                fake = G(zt, cond).detach()
                d_loss = d_hinge_loss(D(z0, cond), D(fake, cond))
                optD.zero_grad(); d_loss.backward(); optD.step()
            # ---- G ----
            zt = torch.randn(B, args.z_dim, device=device)
            g_loss = g_hinge_loss(D(G(zt, cond), cond))
            optG.zero_grad(); g_loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.clip_grad)
            optG.step()
            rd += d_loss.item(); rg += g_loss.item(); seen += 1; step += 1
            if wb:
                wb.log({"loss/d": d_loss.item(), "loss/g": g_loss.item(),
                        "learning_rate": args.lr}, step=step)
            if args.steps and (i + 1) >= args.steps:
                break
        print(f"epoch {epoch+1:>3}/{args.epochs} | d={rd/max(seen,1):.3f} "
              f"g={rg/max(seen,1):.3f} | {time.time()-t0:.1f}s")
        if wb:
            wb.log({"epoch/d": rd/max(seen, 1), "epoch/g": rg/max(seen, 1),
                    "epoch": epoch + 1}, step=step)
            if (epoch + 1) % args.sample_every == 0:
                sp = sample_grid(G, vae, fixed_conds, preview, scale, device,
                                 os.path.join(args.out, "samples_live.png"))
                if sp:
                    wb.log({"samples/live": wb.Image(sp)}, step=step)
                G.train()

    ckpt = os.path.join(args.out, "gan.pt")
    torch.save({"model": G.state_dict(),
                "config": {"z_dim": args.z_dim, "base": args.base, "latent_channels": lc,
                           "latent_hw": hw, "cond_dim": cond_dim, "latent_scale": scale,
                           "image_size": args.image_size, "clip_model": args.clip_model,
                           "clip_pretrained": args.clip_pretrained,
                           "text_cond": args.text_cond, "label_col": args.label_col}},
               ckpt)
    print(f"saved generator → {ckpt}")

    sp = sample_grid(G, vae, fixed_conds, preview, scale, device,
                     os.path.join(args.out, "samples.png"))
    print(f"saved samples → {sp}" if sp else "(torchvision absent — skipped grid)")
    if wb:
        if sp:
            wb.log({"samples": wb.Image(sp)})
        wb.finish()


if __name__ == "__main__":
    main()
