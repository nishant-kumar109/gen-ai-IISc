"""Convolutional VAE for the latent space (from scratch).

Compresses an image into a small **spatial latent** ``z``; the conditional DDPM
then runs *in this latent space* (latent diffusion). Implements the two things
the course cares about: the **reparameterization trick** and the **ELBO**
(reconstruction + β·KL). A VQ-VAE variant is a planned extension.

Default config: 64×64 RGB → latent ``[4, 8, 8]`` (8× spatial downsampling).

This is a ``torch`` module — it runs on Colab / any torch box. Importing it
without torch is safe (the classes just aren't defined); running it prints a
friendly note. Shape self-test on Colab:

    python3 models/vae.py
"""
from __future__ import annotations

import importlib.util

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _norm(ch: int) -> "nn.GroupNorm":
        return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch, eps=1e-6)

    class ResBlock(nn.Module):
        """GroupNorm–SiLU–Conv ×2 with a (projected) residual connection."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                _norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1),
                _norm(out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            return self.block(x) + self.skip(x)

    class Downsample(nn.Module):
        def __init__(self, ch: int) -> None:
            super().__init__()
            self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

        def forward(self, x):
            return self.op(x)

    class Upsample(nn.Module):
        def __init__(self, ch: int) -> None:
            super().__init__()
            self.op = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            return self.op(x)

    class ConvVAE(nn.Module):
        """A small convolutional VAE.

        Args:
            in_channels: image channels (3 for RGB).
            latent_channels: channels of the spatial latent z.
            base: base width of the first conv stage.
            ch_mult: width multiplier per stage; ``len(ch_mult)`` = #down/up-samples.
            beta: weight on the KL term in the ELBO (β-VAE).
        """

        def __init__(
            self,
            in_channels: int = 3,
            latent_channels: int = 4,
            base: int = 64,
            ch_mult: tuple[int, ...] = (1, 2, 4),
            beta: float = 1.0,
        ) -> None:
            super().__init__()
            self.latent_channels = latent_channels
            self.beta = beta

            # ---- encoder ----
            enc: list[nn.Module] = [nn.Conv2d(in_channels, base, 3, padding=1)]
            ch = base
            for mult in ch_mult:
                out = base * mult
                enc += [ResBlock(ch, out), Downsample(out)]
                ch = out
            enc += [_norm(ch), nn.SiLU(),
                    nn.Conv2d(ch, 2 * latent_channels, 3, padding=1)]  # → (μ, logσ²)
            self.encoder = nn.Sequential(*enc)

            # ---- decoder ----
            dec: list[nn.Module] = [nn.Conv2d(latent_channels, ch, 3, padding=1)]
            for mult in reversed(ch_mult):
                out = base * mult
                dec += [ResBlock(ch, out), Upsample(out)]
                ch = out
            dec += [_norm(ch), nn.SiLU(),
                    nn.Conv2d(ch, in_channels, 3, padding=1), nn.Tanh()]  # images in [-1, 1]
            self.decoder = nn.Sequential(*dec)

        # ---- core ops ----
        def encode(self, x):
            mu, logvar = self.encoder(x).chunk(2, dim=1)
            return mu, logvar

        @staticmethod
        def reparameterize(mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_rec = self.decode(z)
            return x_rec, mu, logvar, z

        # ---- ELBO ----
        def loss(self, x, x_rec, mu, logvar):
            # reconstruction: summed over pixels, averaged over batch
            recon = F.mse_loss(x_rec, x, reduction="none").flatten(1).sum(1).mean()
            # KL(q(z|x) ‖ N(0, I)): summed over latent dims, averaged over batch
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).flatten(1).sum(1).mean()
            total = recon + self.beta * kl
            return {"total": total, "recon": recon, "kl": kl}

        @torch.no_grad()
        def encode_to_latent(self, x, sample: bool = False):
            """Return z for the diffusion pipeline (mean by default; sampled if asked)."""
            mu, logvar = self.encode(x)
            return self.reparameterize(mu, logvar) if sample else mu

        @torch.no_grad()
        def sample(self, n: int, latent_hw: int, device="cpu"):
            z = torch.randn(n, self.latent_channels, latent_hw, latent_hw, device=device)
            return self.decode(z)

    def _demo() -> None:
        torch.manual_seed(0)
        vae = ConvVAE()
        x = torch.randn(2, 3, 64, 64)             # a fake batch in [-1, 1]-ish
        x_rec, mu, logvar, z = vae(x)
        losses = vae.loss(x, x_rec, mu, logvar)
        n_params = sum(p.numel() for p in vae.parameters())
        print(f"ConvVAE params : {n_params/1e6:.2f}M")
        print(f"input          : {tuple(x.shape)}")
        print(f"latent z       : {tuple(z.shape)}   (expect [2, 4, 8, 8])")
        print(f"reconstruction : {tuple(x_rec.shape)}")
        print("loss           : " + ", ".join(f"{k}={v.item():.3f}" for k, v in losses.items()))


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("models/vae.py requires PyTorch — run on Colab / a torch environment.")
    else:
        _demo()
