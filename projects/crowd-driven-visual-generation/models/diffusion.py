"""Conditional latent DDPM — the generative core.

A denoising diffusion model that runs **in the VAE latent space** and is
conditioned on the aggregated crowd vector ``c``. Contains:

* ``ConditionalUNet`` — a small U-Net that predicts the noise ε(x_t, t, c). The
  timestep and condition are injected via **FiLM** (feature-wise scale/shift).
  A learned **null** embedding supports classifier-free guidance.
* ``GaussianDiffusion`` — cosine noise schedule, forward ``q_sample``, the
  ε-prediction training loss with **condition-dropout** (for CFG), and a
  **DDIM** sampler with a guidance scale ``w``.

Everything is ``torch``. Importing without torch is safe; running prints a note.
Shape self-test (runs on CPU): ``python3 models/diffusion.py``
"""
from __future__ import annotations

import importlib.util
import math

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ----------------------------------------------------------------------- #
    # building blocks
    # ----------------------------------------------------------------------- #
    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, t):                      # t: [B]
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
            )
            args = t[:, None].float() * freqs[None]
            return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]

    def _norm(ch: int):
        return nn.GroupNorm(min(32, ch), ch, eps=1e-6)

    class FiLMResBlock(nn.Module):
        """ResBlock whose GroupNorm is FiLM-modulated by the (time+cond) embedding."""

        def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
            super().__init__()
            self.norm1 = _norm(in_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.emb = nn.Linear(emb_dim, 2 * out_ch)          # → scale, shift
            self.norm2 = _norm(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x, emb):
            h = self.conv1(F.silu(self.norm1(x)))
            scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
            h = self.conv2(F.silu(h))
            return h + self.skip(x)

    class Downsample(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

        def forward(self, x):
            return self.op(x)

    class Upsample(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.op = nn.Conv2d(ch, ch, 3, padding=1)

        def forward(self, x):
            return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))

    # ----------------------------------------------------------------------- #
    # conditional U-Net (2 resolution levels — sized for an 8×8 latent)
    # ----------------------------------------------------------------------- #
    class ConditionalUNet(nn.Module):
        def __init__(
            self,
            latent_channels: int = 4,
            base: int = 64,
            cond_dim: int = 512,
        ) -> None:
            super().__init__()
            emb_dim = base * 4

            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(base), nn.Linear(base, emb_dim), nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
            )
            self.null_cond = nn.Parameter(torch.randn(emb_dim) * 0.02)  # for CFG

            self.conv_in = nn.Conv2d(latent_channels, base, 3, padding=1)
            # encoder
            self.res_d0 = FiLMResBlock(base, base, emb_dim)          # 8×8, base
            self.down0 = Downsample(base)                            # → 4×4
            self.res_d1 = FiLMResBlock(base, base * 2, emb_dim)      # 4×4, 2·base
            # middle
            self.res_m = FiLMResBlock(base * 2, base * 2, emb_dim)
            # decoder (with skips)
            self.res_u1 = FiLMResBlock(base * 2 + base * 2, base * 2, emb_dim)  # cat(m, d1)
            self.up0 = Upsample(base * 2)                            # → 8×8
            self.res_u0 = FiLMResBlock(base * 2 + base, base, emb_dim)          # cat(up, d0)
            self.out = nn.Sequential(
                _norm(base), nn.SiLU(), nn.Conv2d(base, latent_channels, 3, padding=1)
            )

        def _embed(self, t, cond, drop_mask=None, force_uncond=False):
            t_emb = self.time_mlp(t)
            if force_uncond or cond is None:
                c_emb = self.null_cond.expand(t.shape[0], -1)
            else:
                c_emb = self.cond_mlp(cond)
                if drop_mask is not None:
                    c_emb = torch.where(drop_mask[:, None], self.null_cond, c_emb)
            return t_emb + c_emb

        def forward(self, x, t, cond, drop_mask=None, force_uncond=False):
            emb = self._embed(t, cond, drop_mask, force_uncond)
            h = self.conv_in(x)
            d0 = self.res_d0(h, emb)                 # skip
            d1 = self.res_d1(self.down0(d0), emb)    # skip
            m = self.res_m(d1, emb)
            u = self.res_u1(torch.cat([m, d1], 1), emb)
            u = self.up0(u)
            u = self.res_u0(torch.cat([u, d0], 1), emb)
            return self.out(u)

    # ----------------------------------------------------------------------- #
    # diffusion process
    # ----------------------------------------------------------------------- #
    def _cosine_betas(T: int, s: float = 0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1 - ac[1:] / ac[:-1]
        return betas.clamp(1e-8, 0.999)

    class GaussianDiffusion(nn.Module):
        def __init__(self, model: "ConditionalUNet", timesteps: int = 1000,
                     p_uncond: float = 0.1) -> None:
            super().__init__()
            self.model = model
            self.T = timesteps
            self.p_uncond = p_uncond
            betas = _cosine_betas(timesteps)
            ac = torch.cumprod(1 - betas, dim=0)
            self.register_buffer("betas", betas)
            self.register_buffer("alphas_cumprod", ac)
            self.register_buffer("sqrt_ac", ac.sqrt())
            self.register_buffer("sqrt_1mac", (1 - ac).sqrt())

        def q_sample(self, x0, t, noise):
            return (self.sqrt_ac[t][:, None, None, None] * x0
                    + self.sqrt_1mac[t][:, None, None, None] * noise)

        def p_losses(self, x0, cond, t=None):
            B = x0.shape[0]
            if t is None:
                t = torch.randint(0, self.T, (B,), device=x0.device)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0, t, noise)
            drop = torch.rand(B, device=x0.device) < self.p_uncond      # CFG dropout
            eps = self.model(x_t, t, cond, drop_mask=drop)
            return F.mse_loss(eps, noise)

        @torch.no_grad()
        def ddim_sample(self, shape, cond, w: float = 3.0, steps: int = 50):
            """Deterministic DDIM sampling with classifier-free guidance scale ``w``."""
            device = self.betas.device
            x = torch.randn(shape, device=device)
            times = torch.linspace(self.T - 1, 0, steps).long().to(device)
            B = shape[0]
            for i, t in enumerate(times):
                tb = torch.full((B,), int(t), device=device, dtype=torch.long)
                eps_c = self.model(x, tb, cond, force_uncond=False)
                eps_u = self.model(x, tb, cond, force_uncond=True)
                eps = eps_u + w * (eps_c - eps_u)                       # CFG
                ac_t = self.alphas_cumprod[t]
                ac_prev = self.alphas_cumprod[times[i + 1]] if i + 1 < len(times) \
                    else torch.tensor(1.0, device=device)
                x0 = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
                x0 = x0.clamp(-3, 3)                                    # latent range guard
                x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps    # η = 0
            return x

    def _demo() -> None:
        torch.manual_seed(0)
        unet = ConditionalUNet(latent_channels=4, base=64, cond_dim=512)
        diff = GaussianDiffusion(unet, timesteps=1000)
        x0 = torch.randn(2, 4, 8, 8)              # a fake latent batch
        cond = torch.randn(2, 512)               # a fake aggregated condition
        loss = diff.p_losses(x0, cond)
        n = sum(p.numel() for p in unet.parameters())
        print(f"U-Net params : {n/1e6:.2f}M")
        print(f"train loss    : {loss.item():.4f}  (ε-prediction MSE)")
        samp = diff.ddim_sample((2, 4, 8, 8), cond, w=3.0, steps=8)
        print(f"DDIM sample   : {tuple(samp.shape)}   (expect [2, 4, 8, 8])")


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("models/diffusion.py requires PyTorch — run on Colab / a torch environment.")
    else:
        _demo()
