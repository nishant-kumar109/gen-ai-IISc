"""Image dataset loader for the visual domain.

Yields square, ``[-1, 1]``-normalised RGB tensors ``[3, H, W]``. Sources
(dispatched by ``get_dataset(spec, ...)``):

* ``"synthetic"``           — random smooth images (pure torch; local pipeline tests)
* ``"cifar10"``             — torchvision CIFAR-10 (fast smoke-training)
* a folder path             — every image under it (ImageFolderDataset)
* a HuggingFace dataset id  — e.g. ``"huggan/wikiart"`` (optional; needs `datasets`)

torch is required; torchvision / Pillow / datasets are imported lazily only by
the loaders that need them, so the synthetic path runs with torch alone.
"""
from __future__ import annotations

import importlib.util
import os
import time

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader, Dataset

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # ------------------------------------------------------------------- #
    class SyntheticImageDataset(Dataset):
        """Random low-frequency colour images in [-1, 1]. For pipeline tests."""

        def __init__(self, n: int = 256, image_size: int = 64, seed: int = 0) -> None:
            self.n, self.s = n, image_size
            self.g = torch.Generator().manual_seed(seed)
            # pre-sample a base colour + low-freq field per image
            self.colors = torch.rand(n, 3, 1, 1, generator=self.g) * 2 - 1

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, i: int):
            g = torch.Generator().manual_seed(i)
            low = torch.randn(3, 8, 8, generator=g)
            field = torch.nn.functional.interpolate(
                low[None], size=self.s, mode="bilinear", align_corners=False
            )[0]
            img = (self.colors[i] + 0.5 * field).clamp(-1, 1)
            return img

    # ------------------------------------------------------------------- #
    class ImageFolderDataset(Dataset):
        """Every image file under ``root`` (recursive) → [-1, 1] tensor."""

        def __init__(self, root: str, image_size: int = 64) -> None:
            from PIL import Image  # noqa: F401  (checked at import)

            self.paths = [
                os.path.join(dp, f)
                for dp, _, fs in os.walk(root)
                for f in fs
                if f.lower().endswith(IMG_EXTS)
            ]
            if not self.paths:
                raise FileNotFoundError(f"no images found under {root!r}")
            self.s = image_size

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, i: int):
            from PIL import Image

            img = Image.open(self.paths[i]).convert("RGB")
            # resize shorter side then centre-crop to a square
            w, h = img.size
            scale = self.s / min(w, h)
            img = img.resize((max(self.s, int(w * scale)), max(self.s, int(h * scale))))
            w, h = img.size
            left, top = (w - self.s) // 2, (h - self.s) // 2
            img = img.crop((left, top, left + self.s, top + self.s))
            t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).float()
            t = t.view(self.s, self.s, 3).permute(2, 0, 1) / 255.0
            return t * 2 - 1

    # ------------------------------------------------------------------- #
    def _cifar10(image_size: int, root: str = "./_data"):
        import torchvision
        import torchvision.transforms as T

        tfm = T.Compose([
            T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),   # → [-1, 1]
        ])
        base = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        return _StripLabels(base)

    class _StripLabels(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, _ = self.base[i]
            return x

    class _TensorDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    def _hf_subset(spec, image_size, limit=5000, split="train", image_col="image"):
        """Stream a HuggingFace image dataset and materialise the first ``limit``
        images into memory. Streaming avoids downloading the whole set, and
        materialising lets us do many epochs / shuffling. Returns [-1,1] images."""
        from datasets import load_dataset

        print(f"streaming {spec!r} — first {limit} images...", flush=True)
        try:
            ds = load_dataset(spec, split=split, streaming=True, trust_remote_code=True)
        except TypeError:
            ds = load_dataset(spec, split=split, streaming=True)
        imgs = []
        t0 = time.time()
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            img = ex[image_col].convert("RGB").resize((image_size, image_size))
            t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).float()
            imgs.append(t.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0 * 2 - 1)
            if (i + 1) % 250 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  {i + 1}/{limit} images ({rate:.0f}/s)", flush=True)
        if not imgs:
            raise RuntimeError(f"streamed 0 images from {spec!r}; check image_col={image_col!r}")
        print(f"  materialised {len(imgs)} images", flush=True)
        return _TensorDataset(torch.stack(imgs))

    def stream_with_labels(spec, image_size, limit=15000, split="train",
                           image_col="image", label_col="genre"):
        """Stream a HuggingFace image dataset returning ``(images, label_names)``.

        Used by the text-conditioned diffusion retrain: each image's categorical
        label (WikiArt ``genre``/``style``) is resolved to its *name* so it can be
        embedded by CLIP's text encoder. Returns a ``[N,3,H,W]`` tensor in [-1,1]
        and a list of ``N`` label-name strings. Falls back to ``"artwork"`` if the
        label column is missing.
        """
        from datasets import load_dataset

        print(f"streaming {spec!r} with '{label_col}' labels — first {limit}...", flush=True)
        try:
            ds = load_dataset(spec, split=split, streaming=True, trust_remote_code=True)
        except TypeError:
            ds = load_dataset(spec, split=split, streaming=True)
        feats = getattr(ds, "features", None)
        names = None
        if feats and label_col in feats and hasattr(feats[label_col], "names"):
            names = feats[label_col].names
        else:
            avail = list(feats) if feats else "unknown"
            print(f"  [WARN] label column {label_col!r} not a named ClassLabel "
                  f"(available: {avail}). Labels will be raw values — text conditioning "
                  f"will be weak. Try a different --label-col.", flush=True)
        imgs, labels = [], []
        t0 = time.time()
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            img = ex[image_col].convert("RGB").resize((image_size, image_size))
            t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float()
            imgs.append(t.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0 * 2 - 1)
            lab = ex.get(label_col)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  {i + 1}/{limit} images ({rate:.0f}/s)", flush=True)
            if names is not None and isinstance(lab, int) and 0 <= lab < len(names):
                labels.append(names[lab].replace("_", " "))
            else:
                labels.append(str(lab) if lab is not None else "artwork")
        if not imgs:
            raise RuntimeError(f"streamed 0 images from {spec!r}")
        uniq = sorted(set(labels))
        print(f"  {len(imgs)} images, {len(uniq)} unique '{label_col}' labels; "
              f"examples: {uniq[:6]}", flush=True)
        return torch.stack(imgs), labels

    # ------------------------------------------------------------------- #
    def get_dataset(spec: str = "synthetic", image_size: int = 64,
                    limit: int = 5000, image_col: str = "image", split: str = "train"):
        """Dispatch a dataset by ``spec``. (`limit`/`image_col`/`split` apply only
        to HuggingFace ids.)"""
        if spec == "synthetic":
            return SyntheticImageDataset(image_size=image_size)
        if spec == "cifar10":
            return _cifar10(image_size)
        if os.path.isdir(spec):
            return ImageFolderDataset(spec, image_size)
        # otherwise assume a HuggingFace dataset id
        return _hf_subset(spec, image_size, limit=limit, split=split, image_col=image_col)

    def make_dataloader(spec="synthetic", image_size=64, batch_size=64,
                        shuffle=True, num_workers=2, **kw):
        ds = get_dataset(spec, image_size, **kw)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=True)


def _demo() -> None:
    dl = make_dataloader("synthetic", image_size=64, batch_size=8, num_workers=0)
    x = next(iter(dl))
    print(f"batch shape : {tuple(x.shape)}   (expect [8, 3, 64, 64])")
    print(f"value range : [{x.min():.2f}, {x.max():.2f}]   (expect ~[-1, 1])")
    print(f"dataset size: {len(dl.dataset)}")


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("data/art_dataset.py requires PyTorch — run on Colab.")
    else:
        _demo()
