"""Stage 3 — turn a crowd into an image (inference; the RQ1 aggregation study).

Pipeline (no training for the non-learnable aggregators):

    crowd of words/emojis  →  CLIP text embeddings  →  AGGREGATE → one vector c
                           →  guided DDIM sampling (frozen diffusion) → latent
                           →  VAE decode → the crowd's image

Because the diffusion model was trained on each image's CLIP *image* embedding,
and CLIP shares one space for images and text, the aggregated *text* embedding
of a crowd steers generation. The aggregator (``mean`` vs ``centroid`` …) is the
core comparison: which best turns a diverse crowd into a *coherent* image?

Two study modes (``--mode``):
  * ``themes``    rows = aggregators, cols = themes         (does each theme render?)
  * ``diversity`` rows = aggregators, cols = diversity levels (robustness to noise —
                  the RQ1 punchline: mean-pool → mush, centroid → stays coherent)

Local plumbing test (no CLIP; HashingEncoder stub, random-ish conds):
    python3 sample_crowd.py --vae /tmp/vs7/vae.pt --diffusion /tmp/diff2/diffusion.pt \
        --fake-clip --mode themes --device cpu --out /tmp/crowd.png

Colab (real):
    python3 sample_crowd.py --vae runs/vae/vae.pt --diffusion runs/diffusion/diffusion.pt \
        --mode diversity --theme paradise --out runs/diffusion/crowd_diversity.png
"""
from __future__ import annotations

import argparse
import os

import torch

from data.crowd_simulator import CrowdSimulator
from models.aggregation import aggregate
from models.diffusion import ConditionalUNet, GaussianDiffusion
from models.vae import ConvVAE


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    vae = ConvVAE(latent_channels=c["latent_channels"], base=c["base"],
                  ch_mult=tuple(c["ch_mult"]), beta=c.get("beta", 1.0))
    vae.load_state_dict(ckpt["model"])
    vae = vae.to(device).eval()
    hw = c.get("latent_hw") or (c["image_size"] // (2 ** len(c["ch_mult"])))
    return vae, c["latent_channels"], hw


def load_diffusion(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    unet = ConditionalUNet(latent_channels=c["latent_channels"], base=c["base"],
                           cond_dim=c["cond_dim"], ch_mult=tuple(c["unet_ch_mult"]))
    unet.load_state_dict(ckpt["model"])
    diff = GaussianDiffusion(unet, timesteps=c["timesteps"],
                             p_uncond=c["p_uncond"]).to(device).eval()
    return diff, c


def build_encoder(dcfg, device, fake_clip):
    """Return an encoder with ``encode_texts(list[str]) -> [N, D]`` (D = cond_dim)."""
    if fake_clip:
        from models.encoders import HashingEncoder
        return HashingEncoder(dim=dcfg["cond_dim"])
    from models.encoders import CLIPEncoder
    return CLIPEncoder(model_name=dcfg["clip_model"],
                       pretrained=dcfg["clip_pretrained"], device=device)


def crowd_embeddings(sim, enc, theme, n, diversity):
    """Sample one crowd and encode its responses → ([N, D] embeddings, crowd)."""
    crowd = sim.sample(theme, n=n, diversity=diversity)
    embs = enc.encode_texts([r.value for r in crowd.responses])   # [N, D] unit vectors
    return embs, crowd


def aggregate_cond(agg, embs):
    """Aggregate a crowd's embeddings → one L2-normalised condition vector."""
    kw = {"k": 4, "mode": "dominant"} if agg == "centroid" else {}
    return aggregate(agg, embs, **kw)                    # [D], already L2-normalised


def save_labeled_grid(images, row_labels, col_labels, path, suptitle=None):
    """images: [R][C] tensors in [-1,1] → a labeled R×C figure (matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # fallback: plain grid, no labels
        from torchvision.utils import save_image
        flat = [img for row in images for img in row]
        save_image([(i + 1) / 2 for i in flat], path, nrow=len(col_labels))
        return path

    R, C = len(row_labels), len(col_labels)
    fig, axes = plt.subplots(R, C, figsize=(2.1 * C, 2.1 * R), squeeze=False)
    for r in range(R):
        for c in range(C):
            ax = axes[r][c]
            img = ((images[r][c].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="runs/vae/vae.pt")
    p.add_argument("--diffusion", default="runs/diffusion/diffusion.pt")
    p.add_argument("--mode", default="themes", choices=["themes", "diversity"])
    p.add_argument("--aggregators", default="mean,centroid",
                   help="comma list: mean,centroid (non-learnable)")
    p.add_argument("--themes", default="paradise,fire,love,night,hope",
                   help="themes mode: one column per theme")
    p.add_argument("--theme", default="paradise", help="diversity mode: the theme")
    p.add_argument("--diversities", default="0.1,0.4,0.7",
                   help="diversity mode: one column per level")
    p.add_argument("--n", type=int, default=300, help="crowd size")
    p.add_argument("--diversity", type=float, default=0.2, help="themes mode: fixed level")
    p.add_argument("--guidance", type=float, default=3.0, help="CFG scale w")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-clip", action="store_true", help="HashingEncoder (local test)")
    p.add_argument("--out", default="runs/diffusion/crowd.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vae, lc, hw = load_vae(args.vae, device)
    diff, dcfg = load_diffusion(args.diffusion, device)
    scale = dcfg["latent_scale"]
    enc = build_encoder(dcfg, device, args.fake_clip)
    sim = CrowdSimulator(seed=args.seed)
    aggregators = [a.strip() for a in args.aggregators.split(",")]

    # build the (row=aggregator × col) grid spec
    if args.mode == "themes":
        cols = [t.strip() for t in args.themes.split(",")]
        col_labels = cols
        def theme_div(col):  # noqa: E306
            return col, args.diversity
        suptitle = f"crowd → image · n={args.n}, diversity={args.diversity}, w={args.guidance}"
    else:  # diversity
        cols = [float(d) for d in args.diversities.split(",")]
        col_labels = [f"div={d:g}" for d in cols]
        def theme_div(col):  # noqa: E306
            return args.theme, col
        suptitle = f"crowd → image · theme={args.theme!r}, n={args.n}, w={args.guidance}"

    # one crowd per column, encoded ONCE, then aggregated by each aggregator — so
    # the aggregators are compared on the *same* crowd (a fair RQ1 comparison).
    conds, index = [], []
    for cidx, col in enumerate(cols):
        theme, div = theme_div(col)
        embs, crowd = crowd_embeddings(sim, enc, theme, args.n, div)
        print(f"[{args.mode}:{col_labels[cidx]}] top responses: "
              f"{[w for w, _ in crowd.top(5)]}  (on-theme={crowd.on_theme_fraction():.0%})")
        for r, agg in enumerate(aggregators):
            conds.append(aggregate_cond(agg, embs))
            index.append((r, cidx))

    cond_batch = torch.tensor(conds, dtype=torch.float32, device=device)   # [M, D]
    print(f"sampling {len(conds)} conditions "
          f"({len(aggregators)} aggregators × {len(cols)} cols) — DDIM {args.steps} steps...")
    z = diff.ddim_sample((cond_batch.shape[0], lc, hw, hw), cond_batch,
                         w=args.guidance, steps=args.steps)
    imgs = vae.decode(z / scale).clamp(-1, 1)                              # [M, 3, H, W]

    grid = [[None] * len(cols) for _ in aggregators]
    for (r, cidx), img in zip(index, imgs):
        grid[r][cidx] = img
    path = save_labeled_grid(grid, aggregators, col_labels, args.out, suptitle)
    print(f"saved → {path}")


if __name__ == "__main__":
    main()
