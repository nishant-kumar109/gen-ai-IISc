"""Synthetic crowd-response simulator.

Generates, for a given event *prompt* (theme), a crowd of ``N`` lightweight
audience responses — words, emojis, or doodle labels — with a controllable
**diversity** level. This is the input distribution that the encoding →
aggregation → conditional-diffusion pipeline consumes.

Why simulated? There is no real "audience-response → artwork" dataset, and live
data is private. A simulator lets us (a) train/evaluate the aggregation study
with *known* ground-truth themes, and (b) sweep crowd size ``N`` and diversity
to test how coherence degrades — exactly what RQ1 asks.

Pure standard library (no numpy/torch) so it runs anywhere and is trivially
testable. Determinism is controlled via ``seed``.

Run a demo:
    python3 crowd_simulator.py
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Theme vocabularies: each theme maps to on-theme words / emojis / doodle-labels
# (doodle labels are QuickDraw category names, used later to fetch sketches).
# --------------------------------------------------------------------------- #
THEMES: dict[str, dict[str, list[str]]] = {
    "paradise": {
        "words": ["peace", "calm", "ocean", "sunset", "heaven", "escape",
                  "freedom", "serene", "blue", "waves", "sky", "bliss",
                  "golden", "breeze", "warm"],
        "emojis": ["🌅", "🌊", "☁️", "🕊️", "💙", "✨", "🏝️", "🌴", "🌞", "🍹"],
        "doodles": ["palm tree", "sun", "cloud", "bird", "ocean", "beach",
                    "boat", "flower"],
    },
    "fire": {
        "words": ["energy", "burn", "hot", "power", "wild", "blaze", "spark",
                  "intense", "red", "storm", "loud", "electric", "rage",
                  "bright", "chaos"],
        "emojis": ["🔥", "⚡", "💥", "🌋", "❤️‍🔥", "🎆", "🚨", "😤", "🥵", "🤘"],
        "doodles": ["fire", "lightning", "sun", "star", "volcano", "skull",
                    "campfire", "drums"],
    },
    "love": {
        "words": ["love", "heart", "together", "warm", "tender", "forever",
                  "sweet", "close", "gentle", "romance", "soft", "care",
                  "hug", "dream", "us"],
        "emojis": ["❤️", "💕", "😍", "🥰", "💘", "💞", "🌹", "💫", "💖", "🫶"],
        "doodles": ["heart", "rose", "couple", "ring", "flower", "candle",
                    "letter", "bird"],
    },
    "night": {
        "words": ["night", "stars", "dark", "moon", "quiet", "dream", "deep",
                  "cool", "midnight", "glow", "silent", "sky", "shadow",
                  "cosmic", "still"],
        "emojis": ["🌙", "⭐", "🌌", "✨", "🌃", "🪐", "🦉", "💤", "🌟", "🖤"],
        "doodles": ["moon", "star", "owl", "cloud", "mountain", "campfire",
                    "cat", "house"],
    },
    "hope": {
        "words": ["hope", "rise", "light", "future", "dawn", "believe",
                  "shine", "new", "bright", "together", "change", "dream",
                  "grow", "open", "brave"],
        "emojis": ["🌅", "🌱", "✨", "🕊️", "💛", "🙌", "🌈", "🔆", "🌻", "🤍"],
        "doodles": ["sun", "tree", "bird", "rainbow", "flower", "mountain",
                    "star", "hand"],
    },
}

# generic off-theme / low-effort responses (the "noise" a real crowd produces)
NOISE_WORDS = ["cool", "nice", "wow", "yes", "fun", "good", "great", "lit",
               "vibe", "same", "haha", "ok", "omg", "best", "hey"]
NOISE_EMOJIS = ["😀", "👍", "🎉", "😂", "💯", "🙌", "😎", "🤩", "👀", "🫡"]
NOISE_DOODLES = ["smiley face", "square", "circle", "arrow", "house", "cat"]

MODALITIES = ("word", "emoji", "doodle")


@dataclass(frozen=True)
class Response:
    """A single audience contribution."""
    modality: str            # 'word' | 'emoji' | 'doodle'
    value: str

    def __str__(self) -> str:
        return f"{self.value}"


@dataclass
class Crowd:
    """A sampled crowd of responses for one theme/prompt."""
    theme: str
    n: int
    diversity: float
    responses: list[Response]

    def modality_counts(self) -> Counter:
        return Counter(r.modality for r in self.responses)

    def top(self, k: int = 8) -> list[tuple[str, int]]:
        """Most common response *values* (across all modalities)."""
        return Counter(r.value for r in self.responses).most_common(k)

    def on_theme_fraction(self) -> float:
        """Fraction of responses drawn from this theme's vocabulary."""
        vocab = set(_theme_pool(self.theme))
        hits = sum(1 for r in self.responses if r.value in vocab)
        return hits / max(len(self.responses), 1)


def _theme_pool(theme: str) -> list[str]:
    t = THEMES[theme]
    return t["words"] + t["emojis"] + t["doodles"]


class CrowdSimulator:
    """Samples crowds of audience responses for a theme.

    Args:
        seed: RNG seed for reproducibility.
        modality_mix: probabilities of (word, emoji, doodle) for each response.
        diversity: default fraction of *off-theme* (noise) responses, in [0, 1].
            0.0 → perfectly on-theme (easy, coherent); 1.0 → pure noise (hard).
    """

    def __init__(
        self,
        seed: int = 0,
        modality_mix: tuple[float, float, float] = (0.6, 0.3, 0.1),
        diversity: float = 0.15,
    ) -> None:
        if abs(sum(modality_mix) - 1.0) > 1e-6:
            raise ValueError("modality_mix must sum to 1.0")
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must be in [0, 1]")
        self.rng = random.Random(seed)
        self.modality_mix = modality_mix
        self.diversity = diversity

    def _sample_one(self, theme: str, diversity: float) -> Response:
        modality = self.rng.choices(MODALITIES, weights=self.modality_mix, k=1)[0]
        off_theme = self.rng.random() < diversity
        if modality == "word":
            pool = NOISE_WORDS if off_theme else THEMES[theme]["words"]
        elif modality == "emoji":
            pool = NOISE_EMOJIS if off_theme else THEMES[theme]["emojis"]
        else:  # doodle
            pool = NOISE_DOODLES if off_theme else THEMES[theme]["doodles"]
        return Response(modality=modality, value=self.rng.choice(pool))

    def sample(
        self,
        theme: str,
        n: int = 1000,
        diversity: Optional[float] = None,
    ) -> Crowd:
        """Draw a crowd of ``n`` responses for ``theme``.

        ``diversity`` overrides the instance default for this draw (useful for
        sweeping the coherence-vs-diversity axis in RQ1).
        """
        if theme not in THEMES:
            raise KeyError(f"unknown theme {theme!r}; choices: {list(THEMES)}")
        div = self.diversity if diversity is None else diversity
        responses = [self._sample_one(theme, div) for _ in range(n)]
        return Crowd(theme=theme, n=n, diversity=div, responses=responses)


def available_themes() -> list[str]:
    return list(THEMES)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    sim = CrowdSimulator(seed=42)
    print("Available themes:", available_themes(), "\n")
    for theme in ("paradise", "fire"):
        for div in (0.1, 0.5):
            crowd = sim.sample(theme, n=1000, diversity=div)
            print(f"theme={theme!r}  N={crowd.n}  diversity={div}")
            print(f"  modality mix : {dict(crowd.modality_counts())}")
            print(f"  on-theme frac: {crowd.on_theme_fraction():.2f}")
            print(f"  top responses: {crowd.top(6)}")
            sample = ", ".join(str(r) for r in crowd.responses[:12])
            print(f"  first 12     : {sample}\n")


if __name__ == "__main__":
    _demo()
