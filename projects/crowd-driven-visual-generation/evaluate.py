"""RQ1 evaluation — quantify aggregation strategies for crowd → image.

For each aggregator × diversity (averaged over themes), generate images from many
independent crowds and score them with CLIP:

  * **theme_fidelity** — cos(CLIP_img(generated), CLIP_txt(theme)). Is the image on-theme?
  * **consistency**    — mean pairwise cos among images generated from *independent* crowds
                         of the same theme. Does the aggregator map noisy crowds to a stable
                         target (vs scattering)?

RQ1: as crowd **diversity** (off-theme noise) rises, which aggregator holds fidelity /
consistency best? Hypothesis: cluster-centroid > mean-pool (centroid locks onto the
dominant theme and ignores scattered noise; mean-pool averages everything → drift).

Outputs `metrics.json` + `rq1_fidelity.png` / `rq1_consistency.png` and prints a summary.

    python3 evaluate.py --vae runs/vae/vae.pt --diffusion runs/diffusion/diffusion.pt \
        --repeats 8 --out runs/diffusion/rq1
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from data.crowd_simulator import THEMES, CrowdSimulator
from sample_crowd import aggregate_cond, build_encoder, load_diffusion, load_vae

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def clip_image_embed(enc, x, device):
    """CLIP image embedding of a [-1,1] tensor batch → [B, D] unit vectors."""
    if not hasattr(enc, "model"):                      # fake-clip (local plumbing test)
        f = torch.randn(x.shape[0], 512, device=device)
        return f / f.norm(dim=-1, keepdim=True)
    t = enc._torch
    x = (x + 1) / 2
    x = t.nn.functional.interpolate(x, size=224, mode="bicubic",
                                    align_corners=False, antialias=True).clamp(0, 1)
    mean = t.tensor(_CLIP_MEAN, device=device)[None, :, None, None]
    std = t.tensor(_CLIP_STD, device=device)[None, :, None, None]
    with t.no_grad():
        f = enc.model.encode_image((x - mean) / std)
        return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="runs/vae/vae.pt")
    p.add_argument("--diffusion", default="runs/diffusion/diffusion.pt")
    p.add_argument("--aggregators", default="mean,centroid")
    p.add_argument("--themes", default=",".join(THEMES))
    p.add_argument("--diversities", default="0.1,0.4,0.7")
    p.add_argument("--repeats", type=int, default=8, help="independent crowds per cell")
    p.add_argument("--n", type=int, default=300, help="crowd size")
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--theme-template", default="a {theme} scene",
                   help="text prompt for theme-fidelity (CLIP target)")
    p.add_argument("--gap-file", default=None, help="optional modality-gap correction")
    p.add_argument("--gap-scale", type=float, default=1.0)
    p.add_argument("--fake-clip", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/diffusion/rq1")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    vae, lc, hw = load_vae(args.vae, device)
    diff, dcfg = load_diffusion(args.diffusion, device)
    scale = dcfg["latent_scale"]
    enc = build_encoder(dcfg, device, args.fake_clip)
    sim = CrowdSimulator(seed=args.seed)

    aggregators = [a.strip() for a in args.aggregators.split(",")]
    themes = [t.strip() for t in args.themes.split(",")]
    diversities = [float(d) for d in args.diversities.split(",")]

    gap = None
    if args.gap_file:
        gap = torch.load(args.gap_file, map_location="cpu", weights_only=False)["gap"].numpy()

    # CLIP text embedding of each theme (the fidelity target)
    theme_txt = {}
    for t in themes:
        v = enc.encode_texts([args.theme_template.format(theme=t)])
        theme_txt[t] = torch.tensor(v[0] if hasattr(v, "shape") else v[0],
                                    dtype=torch.float32, device=device)

    rows = []
    for agg in aggregators:
        for div in diversities:
            for theme in themes:
                conds = []
                for _ in range(args.repeats):
                    crowd = sim.sample(theme, n=args.n, diversity=div)
                    embs = enc.encode_texts([r.value for r in crowd.responses])
                    conds.append(aggregate_cond(agg, embs, gap=gap, gap_scale=args.gap_scale))
                cond_batch = torch.tensor(conds, dtype=torch.float32, device=device)
                z = diff.ddim_sample((cond_batch.shape[0], lc, hw, hw), cond_batch,
                                     w=args.guidance, steps=args.steps)
                imgs = vae.decode(z / scale).clamp(-1, 1)
                ie = clip_image_embed(enc, imgs, device)              # [K, D] unit
                fid = float((ie @ theme_txt[theme]).mean())           # cos, both unit
                K = ie.shape[0]
                simm = ie @ ie.T                                      # [K, K] cosines
                consistency = float((simm.sum() - K) / max(K * (K - 1), 1))
                rows.append({"aggregator": agg, "theme": theme, "diversity": div,
                             "theme_fidelity": fid, "consistency": consistency})
            print(f"{agg:>8} | div={div:<4} done ({len(themes)} themes × {args.repeats} crowds)")

    # ---- summarise: average over themes → per (aggregator, diversity) ----
    summary = {}
    for agg in aggregators:
        summary[agg] = {}
        for div in diversities:
            sel = [r for r in rows if r["aggregator"] == agg and r["diversity"] == div]
            summary[agg][div] = {
                "theme_fidelity": sum(r["theme_fidelity"] for r in sel) / len(sel),
                "consistency": sum(r["consistency"] for r in sel) / len(sel),
            }

    json.dump({"rows": rows, "summary": summary, "config": vars(args)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    # ---- print table ----
    print("\n=== RQ1 summary (higher = better; averaged over themes) ===")
    print(f"{'aggregator':>10} {'diversity':>10} {'fidelity':>10} {'consistency':>12}")
    for agg in aggregators:
        for div in diversities:
            s = summary[agg][div]
            print(f"{agg:>10} {div:>10} {s['theme_fidelity']:>10.4f} {s['consistency']:>12.4f}")

    # verdict at the highest diversity (robustness to noise)
    hi = max(diversities)
    if len(aggregators) >= 2:
        best = max(aggregators, key=lambda a: summary[a][hi]["theme_fidelity"])
        print(f"\nAt diversity={hi} (noisiest crowd), highest theme-fidelity: "
              f"**{best}** ({summary[best][hi]['theme_fidelity']:.4f}).")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric, fname, ylab in [("theme_fidelity", "rq1_fidelity.png", "CLIP theme-fidelity"),
                                    ("consistency", "rq1_consistency.png", "output consistency")]:
            plt.figure(figsize=(5, 3.5))
            for agg in aggregators:
                ys = [summary[agg][d][metric] for d in diversities]
                plt.plot(diversities, ys, marker="o", label=agg)
            plt.xlabel("crowd diversity (off-theme noise)"); plt.ylabel(ylab)
            plt.title(f"RQ1: {ylab} vs diversity"); plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(args.out, fname), dpi=130)
            plt.close()
        print(f"\nsaved metrics + plots → {args.out}/")
    except Exception as e:
        print(f"(plots skipped: {e}) — metrics.json saved to {args.out}/")


if __name__ == "__main__":
    main()
