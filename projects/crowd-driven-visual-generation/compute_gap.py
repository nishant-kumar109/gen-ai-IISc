"""Measure the CLIP **modality gap** — the offset between the image-embedding
cloud and the text-embedding cloud in CLIP's shared space.

The diffusion model (Stage 2) was trained on CLIP *image* embeddings, but a crowd
condition (Stage 3) is a CLIP *text* embedding. Those clouds don't coincide, so
text conditions are out-of-distribution for the model. This computes

    gap = mean(image embeddings)  −  mean(text embeddings)

so ``sample_crowd.py --gap-file`` can translate a crowd's text condition toward
the image manifold: ``cond = normalize(aggregate(text) + gap)``.

Cheap: one pass over ~2000 images + the crowd vocabulary. No retraining.

    python3 compute_gap.py --diffusion runs/diffusion/diffusion.pt \
        --dataset huggan/wikiart --limit 2000 --out runs/diffusion/gap.pt
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from data.art_dataset import make_dataloader
from data.crowd_simulator import THEMES, CrowdSimulator

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def image_embed(model, x, device):
    x = (x + 1) / 2
    x = torch.nn.functional.interpolate(x, size=224, mode="bicubic",
                                        align_corners=False, antialias=True).clamp(0, 1)
    mean = torch.tensor(_CLIP_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(_CLIP_STD, device=device)[None, :, None, None]
    f = model.encode_image((x - mean) / std)
    return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion", default="runs/diffusion/diffusion.pt",
                   help="read the matching CLIP model/pretrained from its config")
    p.add_argument("--clip-model", default=None, help="override (else from checkpoint)")
    p.add_argument("--clip-pretrained", default=None)
    p.add_argument("--dataset", default="huggan/wikiart")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--image-col", default="image")
    p.add_argument("--n-text", type=int, default=400, help="crowd size per (theme, div)")
    p.add_argument("--out", default="runs/diffusion/gap.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # CLIP settings that MATCH training (from the diffusion checkpoint)
    cm, cp = args.clip_model, args.clip_pretrained
    if (cm is None or cp is None) and os.path.exists(args.diffusion):
        dc = torch.load(args.diffusion, map_location="cpu", weights_only=False)["config"]
        cm = cm or dc.get("clip_model", "ViT-B-32")
        cp = cp or dc.get("clip_pretrained", "laion2b_s34b_b79k")
    cm, cp = cm or "ViT-B-32", cp or "laion2b_s34b_b79k"

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(cm, pretrained=cp)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cm)
    print(f"CLIP {cm} ({cp})")

    # ---- image cloud mean ----
    dl = make_dataloader(args.dataset, image_size=args.image_size, batch_size=args.batch,
                         shuffle=False, num_workers=2, limit=args.limit,
                         image_col=args.image_col)
    t0, acc, n = time.time(), None, 0
    for x in dl:
        f = image_embed(model, x.to(device), device)         # [B, D] unit
        acc = f.sum(0) if acc is None else acc + f.sum(0)
        n += f.shape[0]
    img_mean = (acc / n).cpu()
    print(f"  image mean over {n} images in {time.time()-t0:.1f}s")

    # ---- text cloud mean (over the crowd vocabulary the simulator produces) ----
    sim = CrowdSimulator(seed=0)
    texts = []
    for theme in THEMES:
        for div in (0.2, 0.5):
            texts += [r.value for r in sim.sample(theme, n=args.n_text, diversity=div).responses]
    tacc, tn = None, 0
    for i in range(0, len(texts), 512):
        chunk = texts[i:i + 512]
        tok = tokenizer(chunk).to(device)
        f = model.encode_text(tok)
        f = f / f.norm(dim=-1, keepdim=True)
        tacc = f.sum(0) if tacc is None else tacc + f.sum(0)
        tn += f.shape[0]
    txt_mean = (tacc / tn).cpu()
    print(f"  text mean over {tn} responses")

    gap = img_mean - txt_mean
    torch.save({"gap": gap, "img_mean": img_mean, "txt_mean": txt_mean,
                "clip_model": cm, "clip_pretrained": cp,
                "gap_norm": float(gap.norm())}, args.out)
    # a unit text embedding has norm 1; report the gap size relative to that
    print(f"gap ‖·‖ = {float(gap.norm()):.3f}  (image & text clouds are this far apart)")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
