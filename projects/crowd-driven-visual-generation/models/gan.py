"""Conditional GAN baseline — an alternative generator to the diffusion model.

Same setup as the diffusion core (fair comparison): generates in the frozen VAE
**latent space**, conditioned on a CLIP embedding, so inference is identical —
``G(noise, aggregated_crowd) -> latent -> VAE.decode -> image``.

* ``Generator``     — (noise ⊕ cond) → latent ``[B, C, hw, hw]`` via transposed convs.
* ``Discriminator`` — a **projection** critic (Miyato & Koyama, 2018): unconditional
  score + ⟨features, embed(cond)⟩. Trained with the **hinge** GAN loss (stable-ish).

torch only. Shape self-test: ``python3 models/gan.py``.
"""
from __future__ import annotations

import importlib.util

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _sn(m):                                    # spectral norm (D stability)
        return nn.utils.spectral_norm(m)

    class Generator(nn.Module):
        """(z ⊕ cond) → latent. Starts at 4×4, upsamples to ``hw`` (16 → two ×2)."""

        def __init__(self, z_dim: int = 128, cond_dim: int = 512,
                     latent_channels: int = 4, hw: int = 16, base: int = 128) -> None:
            super().__init__()
            self.z_dim, self.hw = z_dim, hw
            self.start = hw // 4                    # 16 → 4×4 start, two upsamples
            self.fc = nn.Linear(z_dim + cond_dim, base * 4 * self.start * self.start)
            ch = base * 4
            layers = []
            while self.start * (2 ** (len(layers) // 3)) < hw:
                out = ch // 2
                layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                           nn.Conv2d(ch, out, 3, padding=1),
                           nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
                ch = out
            self.up = nn.Sequential(*layers)
            self.out = nn.Conv2d(ch, latent_channels, 3, padding=1)

        def forward(self, z, cond):
            h = self.fc(torch.cat([z, cond], dim=1))
            h = h.view(z.shape[0], -1, self.start, self.start)
            return self.out(self.up(h))             # [B, C, hw, hw]

        @torch.no_grad()
        def sample(self, cond, device=None):
            device = device or cond.device
            z = torch.randn(cond.shape[0], self.z_dim, device=device)
            return self(z, cond)

    class Discriminator(nn.Module):
        """Projection critic: down-conv → pooled features → score + ⟨feat, embed(cond)⟩."""

        def __init__(self, cond_dim: int = 512, latent_channels: int = 4,
                     hw: int = 16, base: int = 128) -> None:
            super().__init__()
            ch, cur, layers = base, hw, []
            layers += [_sn(nn.Conv2d(latent_channels, ch, 3, padding=1)),
                       nn.LeakyReLU(0.2, inplace=True)]
            while cur > 4:                          # 16 → 8 → 4
                layers += [_sn(nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1)),
                           nn.LeakyReLU(0.2, inplace=True)]
                ch, cur = ch * 2, cur // 2
            self.body = nn.Sequential(*layers)
            self.feat_dim = ch
            self.score = _sn(nn.Linear(ch, 1))      # unconditional score
            self.cond_embed = _sn(nn.Linear(cond_dim, ch))   # projection

        def forward(self, x, cond):
            h = self.body(x).mean(dim=(2, 3))       # global-avg-pool → [B, feat_dim]
            out = self.score(h)                     # [B, 1]
            proj = (h * self.cond_embed(cond)).sum(1, keepdim=True)   # ⟨feat, embed(c)⟩
            return (out + proj).squeeze(1)          # [B]

    def d_hinge_loss(d_real, d_fake):
        return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

    def g_hinge_loss(d_fake):
        return -d_fake.mean()

    def _demo():
        torch.manual_seed(0)
        G = Generator(z_dim=128, cond_dim=512, latent_channels=4, hw=16, base=128)
        D = Discriminator(cond_dim=512, latent_channels=4, hw=16, base=128)
        cond = torch.randn(2, 512)
        z = torch.randn(2, 128)
        fake = G(z, cond)
        dr, df = D(torch.randn(2, 4, 16, 16), cond), D(fake, cond)
        print(f"G params : {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
        print(f"D params : {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
        print(f"fake latent : {tuple(fake.shape)}   (expect [2, 4, 16, 16])")
        print(f"D(real),D(fake): {tuple(dr.shape)},{tuple(df.shape)}   (expect [2],[2])")
        print(f"D hinge={d_hinge_loss(dr, df).item():.3f}  G hinge={g_hinge_loss(df).item():.3f}")


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("models/gan.py requires PyTorch — run on Colab / a torch environment.")
    else:
        _demo()
