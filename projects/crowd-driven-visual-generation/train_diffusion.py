"""Train the conditional latent diffusion model (Stage 2 — the generative core).

Freezes the Stage-1 VAE, encodes every image to its ``[C, h, h]`` latent, and
computes each image's **CLIP embedding** as the conditioning vector. A
``ConditionalUNet`` is then trained (ε-prediction, classifier-free guidance) to
denoise latents *given that embedding*.

Why condition on the image's *own* CLIP embedding? CLIP puts images and text in
**one** space, so a model trained on image embeddings can, at inference, be
steered by a **crowd** embedding aggregated from words/emojis (the RQ1 novelty).
Aggregation is applied only at inference — never here.

Latents are pre-encoded once and cached (the VAE + CLIP are frozen), then scaled
to ~unit variance (DDPM assumes that). The scale is saved in the checkpoint.

Examples
--------
Local plumbing test (no CLIP / no download):
    python3 train_diffusion.py --vae /tmp/vs7/vae.pt --dataset synthetic \
        --fake-clip --epochs 1 --steps 5 --batch 8 --device cpu

Colab (real):
    python3 train_diffusion.py --vae runs/vae/vae.pt --dataset huggan/wikiart \
        --limit 5000 --epochs 60 --batch 128 --out runs/diffusion --wandb
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F

from data.art_dataset import make_dataloader
from models.diffusion import ConditionalUNet, GaussianDiffusion
from models.vae import ConvVAE

# CLIP's expected input normalisation (OpenAI stats, used by open_clip too)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vae(path: str, device: str):
    """Load a frozen ConvVAE from a Stage-1 checkpoint (architecture from config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vae = ConvVAE(latent_channels=cfg["latent_channels"], base=cfg["base"],
                  ch_mult=tuple(cfg["ch_mult"]), beta=cfg.get("beta", 1.0))
    vae.load_state_dict(ckpt["model"])
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, cfg


def clip_embed(clip_model, x, device):
    """Embed images ``x`` (``[B,3,H,W]`` in [-1,1]) with CLIP → L2-normalised [B,D]."""
    x = (x + 1) / 2                                            # → [0,1]
    x = F.interpolate(x, size=224, mode="bicubic", align_corners=False,
                      antialias=True).clamp(0, 1)
    mean = torch.tensor(_CLIP_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(_CLIP_STD, device=device)[None, :, None, None]
    feats = clip_model.encode_image((x - mean) / std)
    return feats / feats.norm(dim=-1, keepdim=True)


def clip_text_embed(clip_model, tokenizer, prompts, device):
    """CLIP text embedding of a list of prompts → [len, D] unit vectors."""
    tok = tokenizer(prompts).to(device)
    f = clip_model.encode_text(tok)
    return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def precompute_text(vae, clip_model, tokenizer, spec, image_size, limit, image_col,
                    label_col, batch, device, keep_preview=8):
    """Text-conditioned precompute: cache latents + per-image CLIP **image** and
    **text**(label) embeddings, so training can mix the two condition types."""
    from data.art_dataset import stream_with_labels

    images, labels = stream_with_labels(spec, image_size, limit=limit,
                                        image_col=image_col, label_col=label_col)
    preview = images[:keep_preview].clone()
    lats, img_conds = [], []
    for i in range(0, len(images), batch):
        xb = images[i:i + batch].to(device)
        lats.append(vae.encode_to_latent(xb).cpu())
        img_conds.append(clip_embed(clip_model, xb, device).cpu())
    latents = torch.cat(lats)
    image_conds = torch.cat(img_conds)
    # embed each unique label once (as a text prompt), then map back per image
    uniq = sorted(set(labels))
    emb = clip_text_embed(clip_model, tokenizer, [f"a {s} painting" for s in uniq], device)
    idx = {s: i for i, s in enumerate(uniq)}
    text_conds = torch.stack([emb[idx[l]] for l in labels]).cpu()
    print(f"  cached {len(latents)} latents + image/text conds "
          f"({len(uniq)} unique labels)")
    return latents, image_conds, text_conds, preview


@torch.no_grad()
def precompute(vae, clip_model, dl, device, fake_clip, keep_preview=8):
    """Encode the whole dataset once → (latents, conds). Returns preview images too."""
    lats, conds, preview = [], [], None
    t0 = time.time()
    for x in dl:
        x = x.to(device)
        if preview is None:
            preview = x[:keep_preview].cpu()
        z = vae.encode_to_latent(x)                           # [B,C,h,h] (posterior mean)
        if fake_clip:                                         # local plumbing: random unit conds
            c = torch.randn(x.shape[0], 512, device=device)
            c = c / c.norm(dim=-1, keepdim=True)
        else:
            c = clip_embed(clip_model, x, device)
        lats.append(z.cpu())
        conds.append(c.cpu())
    latents = torch.cat(lats)
    conditions = torch.cat(conds)
    print(f"  encoded {len(latents)} latents {tuple(latents.shape[1:])} + "
          f"conds {tuple(conditions.shape[1:])} in {time.time()-t0:.1f}s")
    return latents, conditions, preview


@torch.no_grad()
def sample_grid(diffusion, vae, conds, preview, scale, lc, hw, w, steps, device, path):
    """Generate from conds, decode, and save a grid: top = real, bottom = generated."""
    diffusion.eval()
    k = conds.shape[0]
    z = diffusion.ddim_sample((k, lc, hw, hw), conds.to(device), w=w, steps=steps)
    imgs = vae.decode(z / scale).clamp(-1, 1)                 # un-scale before decode
    grid = torch.cat([preview.to(device), imgs], dim=0)       # [2k, 3, H, W]
    grid = (grid.cpu() + 1) / 2
    try:
        from torchvision.utils import save_image
        save_image(grid, path, nrow=k)
        return path
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="runs/vae/vae.pt", help="Stage-1 VAE checkpoint")
    p.add_argument("--dataset", default="synthetic",
                   help="synthetic | cifar10 | <folder> | <hf-dataset-id>")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps", type=int, default=0, help="cap steps/epoch (0 = full)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64, help="U-Net base width")
    p.add_argument("--unet-ch-mult", default="1,2,4",
                   help="U-Net depth (its own; #entries-1 = down/up-samples). "
                        "'1,2,4' fits a 16×16 latent (16→8→4).")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--p-uncond", type=float, default=0.1, help="CFG cond-dropout prob")
    p.add_argument("--guidance", type=float, default=3.0, help="CFG scale w (sampling)")
    p.add_argument("--sample-steps", type=int, default=50, help="DDIM steps (sampling)")
    p.add_argument("--sample-every", type=int, default=5, help="log samples every N epochs")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--text-cond", action="store_true",
                   help="ALSO condition on CLIP *text* embeddings of each image's label "
                        "(bridges the modality gap so crowd text works at inference)")
    p.add_argument("--p-text", type=float, default=0.5,
                   help="fraction of steps conditioned on the text (vs image) embedding")
    p.add_argument("--label-col", default="genre",
                   help="dataset label column to embed as text (WikiArt: genre | style)")
    p.add_argument("--fake-clip", action="store_true",
                   help="skip CLIP, use random conds (local plumbing test only)")
    p.add_argument("--clip-grad", type=float, default=1.0, help="max grad norm (0 = off)")
    p.add_argument("--out", default="runs/diffusion")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=5000, help="max images (HF datasets)")
    p.add_argument("--image-col", default="image", help="image column (HF datasets)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="crowd-driven-visual-generation")
    args = p.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    unet_ch_mult = tuple(int(m) for m in args.unet_ch_mult.split(","))
    print(f"device={device}  vae={args.vae}  dataset={args.dataset}  out={args.out}")

    # ---- frozen Stage-1 VAE ----
    vae, vcfg = load_vae(args.vae, device)
    lc = vcfg["latent_channels"]
    hw = vcfg.get("latent_hw") or (
        vcfg["image_size"] // (2 ** len(vcfg["ch_mult"])))   # fallback for old ckpts
    print(f"loaded VAE: latent [{lc}, {hw}, {hw}]  (frozen)")

    # ---- frozen CLIP (the conditioning encoder) ----
    clip_model, tokenizer = None, None
    if not args.fake_clip:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrained)
        clip_model = clip_model.to(device).eval()
        for pp in clip_model.parameters():
            pp.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(args.clip_model)
        print(f"loaded CLIP {args.clip_model} ({args.clip_pretrained})")

    # ---- W&B ----
    wb = None
    if args.wandb:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=f"diffusion-w{args.guidance:g}",
                    config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            wb = None

    # ---- pre-encode the dataset once (VAE + CLIP frozen) ----
    print("pre-encoding latents + conditions...")
    image_conds = text_conds = conds = None
    if args.text_cond:
        if args.fake_clip:                                   # local plumbing only
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

    # scale latents to ~unit std (DDPM assumption); save the scale for decoding
    scale = 1.0 / (latents.std().item() + 1e-8)
    latents = latents * scale
    print(f"latent scale={scale:.4f}  cond_dim={cond_dim}"
          + (f"  | text-cond mix p={args.p_text}" if args.text_cond else ""))

    if args.text_cond:
        ds = torch.utils.data.TensorDataset(latents, image_conds, text_conds)
    else:
        ds = torch.utils.data.TensorDataset(latents, conds)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                         drop_last=True)

    # ---- model ----
    unet = ConditionalUNet(latent_channels=lc, base=args.base, cond_dim=cond_dim,
                           ch_mult=unet_ch_mult).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=args.timesteps,
                                  p_uncond=args.p_uncond).to(device)
    opt = torch.optim.Adam(unet.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"ConditionalUNet: {n_params/1e6:.2f}M params | batches/epoch={len(loader)}")
    if wb:
        wb.watch(unet, log="all", log_freq=100)

    # a fixed set of conditions for consistent sample grids across epochs.
    # In text-cond mode, visualise from the *text* (label) embeddings — that's the
    # capability we're adding, so the grid shows text-conditioned generation.
    n_prev = preview.shape[0]
    fixed_conds = (text_conds if args.text_cond else conds)[:n_prev].clone()

    step = 0
    for epoch in range(args.epochs):
        unet.train()
        t0, running, seen = time.time(), 0.0, 0
        for i, batch in enumerate(loader):
            if args.text_cond:
                z, ic, tc = (b.to(device) for b in batch)
                # per-sample coin flip: condition on text vs image embedding
                use_text = (torch.rand(z.shape[0], device=device) < args.p_text)[:, None]
                c = torch.where(use_text, tc, ic)
            else:
                z, c = batch[0].to(device), batch[1].to(device)
            loss = diffusion.p_losses(z, c)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                unet.parameters(), args.clip_grad if args.clip_grad > 0 else 1e9)
            opt.step()
            running += loss.item()
            seen += 1
            step += 1
            if wb:
                wb.log({"loss/eps_mse": loss.item(), "grad_norm": float(gn),
                        "learning_rate": opt.param_groups[0]["lr"]}, step=step)
            if args.steps and (i + 1) >= args.steps:
                break
        avg = running / max(seen, 1)
        print(f"epoch {epoch+1:>3}/{args.epochs} | eps_mse={avg:.4f} | {time.time()-t0:.1f}s")
        if wb:
            wb.log({"epoch/eps_mse": avg, "epoch": epoch + 1}, step=step)
            if (epoch + 1) % args.sample_every == 0:
                sp = sample_grid(diffusion, vae, fixed_conds, preview, scale, lc, hw,
                                 args.guidance, args.sample_steps, device,
                                 os.path.join(args.out, "samples_live.png"))
                if sp:
                    wb.log({"samples/live": wb.Image(sp)}, step=step)
                unet.train()

    ckpt = os.path.join(args.out, "diffusion.pt")
    torch.save({"model": unet.state_dict(),
                "config": {"base": args.base, "latent_channels": lc, "latent_hw": hw,
                           "cond_dim": cond_dim, "unet_ch_mult": list(unet_ch_mult),
                           "timesteps": args.timesteps, "p_uncond": args.p_uncond,
                           "latent_scale": scale, "image_size": args.image_size,
                           "clip_model": args.clip_model,
                           "clip_pretrained": args.clip_pretrained,
                           "text_cond": args.text_cond, "p_text": args.p_text,
                           "label_col": args.label_col}},
               ckpt)
    print(f"saved checkpoint → {ckpt}")

    sp = sample_grid(diffusion, vae, fixed_conds, preview, scale, lc, hw,
                     args.guidance, args.sample_steps, device,
                     os.path.join(args.out, "samples.png"))
    print(f"saved samples → {sp}" if sp else "(torchvision absent — skipped sample grid)")
    if wb:
        if sp:
            wb.log({"samples": wb.Image(sp)})
        wb.finish()


if __name__ == "__main__":
    main()
