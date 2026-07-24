"""Train the learnable aggregators (Deep-Sets, attention-PMA) for the RQ1 study.

`mean` and `centroid` are fixed heuristics; these two are *learned*. Objective
(self-supervised, via the crowd simulator + CLIP — no diffusion in the loop):

    given a crowd's response embeddings (noisy: on- + off-theme mix),
    produce a vector close to the crowd's CLEAN theme embedding.

    loss = 1 - cos( aggregator(crowd_embeddings),  CLIP_text("a {theme} scene") )

This rewards **robust aggregation** — extract the dominant theme, ignore scattered
off-theme noise — the exact property RQ1 measures. Trained at the same crowd size
(`--n`) used at evaluation so the (size-sensitive) Deep-Sets sum matches.

    python3 train_aggregators.py --diffusion runs/diffusion_v2/diffusion.pt \
        --steps 1500 --out runs/diffusion_v2/aggregators.pt
"""
from __future__ import annotations

import argparse
import os
import random

import torch

from data.crowd_simulator import THEMES, CrowdSimulator
from models.aggregation import AttentionPooling, DeepSets


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_embedder(dcfg, device, fake_clip):
    """Return embed(list[str]) -> [len, D] unit-vector tensor on device."""
    if fake_clip:
        from models.encoders import HashingEncoder
        enc = HashingEncoder(dim=dcfg["cond_dim"])

        def embed(words):
            return torch.tensor(enc.encode_texts(list(words)),
                                dtype=torch.float32, device=device)
        return embed

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        dcfg["clip_model"], pretrained=dcfg["clip_pretrained"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = open_clip.get_tokenizer(dcfg["clip_model"])

    @torch.no_grad()
    def embed(words):
        words = list(words)
        out = []
        for i in range(0, len(words), 256):
            t = tok(words[i:i + 256]).to(device)
            f = model.encode_text(t)
            out.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(out)
    return embed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion", default="runs/diffusion_v2/diffusion.pt",
                   help="read cond_dim + matching CLIP model from its config")
    p.add_argument("--aggregators", default="deepsets,attention")
    p.add_argument("--themes", default=",".join(THEMES))
    p.add_argument("--n", type=int, default=300, help="crowd size (match evaluation)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256, help="Deep-Sets hidden width")
    p.add_argument("--heads", type=int, default=4, help="attention heads (PMA)")
    p.add_argument("--fake-clip", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/diffusion_v2/aggregators.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    dcfg = torch.load(args.diffusion, map_location="cpu", weights_only=False)["config"]
    cond_dim = dcfg["cond_dim"]
    embed = build_embedder(dcfg, device, args.fake_clip)
    sim = CrowdSimulator(seed=args.seed)
    themes = [t.strip() for t in args.themes.split(",")]

    # clean per-theme target vectors
    theme_targets = {t: embed([f"a {t} scene"])[0] for t in themes}

    # collect the full response vocabulary once → matrix V + word→index (fast lookup)
    vocab = set()
    for t in themes:
        for _ in range(20):
            vocab.update(r.value for r in sim.sample(t, n=500, diversity=0.9).responses)
    vocab = sorted(vocab)
    V = embed(vocab)                                    # [vocab, D]
    w2i = {w: i for i, w in enumerate(vocab)}
    print(f"vocab={len(vocab)} words | cond_dim={cond_dim} | themes={themes}")

    def sample_batch(B, n):
        idxs, tg = [], []
        for _ in range(B):
            th = random.choice(themes)
            dv = random.uniform(0.0, 0.8)
            words = [r.value for r in sim.sample(th, n=n, diversity=dv).responses]
            idxs.append([w2i.get(w, 0) for w in words])
            tg.append(theme_targets[th])
        return V[torch.tensor(idxs, device=device)], torch.stack(tg)   # [B,n,D], [B,D]

    saved = {"config": {"cond_dim": cond_dim, "hidden": args.hidden,
                        "heads": args.heads, "n_train": args.n}}
    for name in [a.strip() for a in args.aggregators.split(",")]:
        if name == "deepsets":
            agg = DeepSets.build(cond_dim, args.hidden, cond_dim).to(device)
        elif name == "attention":
            agg = AttentionPooling.build(cond_dim, cond_dim, args.heads).to(device)
        else:
            raise ValueError(f"unknown learnable aggregator {name!r}")
        opt = torch.optim.Adam(agg.parameters(), lr=args.lr)
        agg.train()
        last = 0.0
        for step in range(args.steps):
            x, tgt = sample_batch(args.batch, args.n)
            c = agg(x)                                  # [B, D] (unit-norm)
            loss = (1 - (c * tgt).sum(-1)).mean()       # cosine loss to clean theme
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
            if (step + 1) % 200 == 0:
                print(f"[{name}] step {step+1}/{args.steps}  cos-loss={last:.4f}")
        saved[name] = agg.state_dict()
        print(f"[{name}] done — final cos-loss={last:.4f} "
              f"(cos-sim to clean theme = {1-last:.3f})")

    torch.save(saved, args.out)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
