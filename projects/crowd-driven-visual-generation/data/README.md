# data/

Data preparation and the **crowd-response simulator**.

## ✅ `crowd_simulator.py` (implemented)
Pure-Python (stdlib only) simulator of audience responses. For a given theme it samples `N`
responses across **word / emoji / doodle** modalities, with a **`diversity`** dial that controls
the fraction of off-theme "noise" — i.e. how coherent vs. scattered the crowd is (the RQ1 axis).

```python
from crowd_simulator import CrowdSimulator
sim = CrowdSimulator(seed=42)
crowd = sim.sample("paradise", n=1000, diversity=0.1)   # 0=coherent … 1=pure noise
crowd.modality_counts(); crowd.top(8); crowd.on_theme_fraction()
```
Run the demo: `python3 crowd_simulator.py`. Themes: paradise, fire, love, night, hope.

---
### Still to add

- **Crowd simulator** — generates theme-related sets of words / emojis / doodles per "song," with controllable size (N) and diversity.
- **Doodles** — QuickDraw (Google) samples.
- **Words / emojis** — public word & emoji embedding sources.
- **Visual domain** — a public art dataset (e.g., WikiArt abstract subset / curated abstract-poster set) for training the VAE + diffusion; held-out split reserved for FID.

(Scripts + downloaded/generated data go here.)
