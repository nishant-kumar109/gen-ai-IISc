# Crowd-Driven Visual Generation with Conditional Diffusion Models

*Extended project summary & vision. The formal template submission is in [PROPOSAL.md](PROPOSAL.md) / `project-proposal-nishant.pdf`; this document gives the fuller narrative, architecture, and roadmap.*

---

## Project Summary

Live concerts and large-scale events are inherently collective experiences, yet audience participation remains largely passive — limited to cheering, flashlights, or singing along. While recent advances in generative AI have enabled highly personalized content generation, relatively little attention has been given to **collective generation**, where a single visual output is influenced by the interactions of many participants.

This project investigates **crowd-driven visual generation**, where lightweight audience interactions (e.g., text, emojis, doodles, or optional images) are aggregated into a unified conditioning representation that guides a conditional diffusion model to synthesize a coherent visual reflecting the audience's shared experience. Rather than generating content from a single prompt, the project explores a **many-to-one conditioning paradigm**, where multiple independent audience inputs collectively influence a single evolving visual narrative.

The generated visuals are intended to be displayed during live performances, transforming audience participation into an interactive and AI-assisted storytelling experience.

## Problem Statement

Most existing generative image models are designed around **one-to-one generation**, where a single prompt or image conditions the output. This project explores a different setting:

> **How can multiple independent audience interactions be represented and aggregated into a unified conditioning signal for generating a coherent visual representation of a shared live experience?**

## Objectives

- Design a pipeline for aggregating multimodal audience interactions into a unified latent representation.
- Investigate conditional diffusion models for crowd-driven visual generation.
- Compare different aggregation and conditioning strategies.
- Study the trade-off between semantic consistency, visual quality, and diversity.
- Prototype an interactive system suitable for live events.

## Product Vision

Imagine a live concert where, during selected songs, attendees are invited to participate through the event application. Example prompts:

- *Describe this moment in one word.*
- *Choose the emoji that best represents your feeling.*
- *Sketch a quick doodle.*
- *Share your favourite lyric.*

Instead of displaying raw audience responses, the system continuously aggregates these interactions and generates a single evolving visual that represents the collective audience sentiment. The visual becomes a dynamic part of the stage experience — a shared artistic representation of the crowd's participation.

## System Architecture

```text
                     Audience Interaction
          ┌──────────────┬──────────────┐
          │              │              │
        Text           Emoji      Doodle / Image
          │              │              │
          └──────────────┴──────────────┘
                         │
              Multimodal Feature Extraction
            (Sentence embeddings / CLIP features)
                         │
                         ▼
              Semantic & Latent Aggregation
                         │
                Event Metadata Integration
             (Artist, Song, Genre, Venue, …)
                         │
                         ▼
            Unified Conditioning Representation
                         │
                         ▼
              Conditional Diffusion Model (DDPM)
                 + Classifier-Free Guidance
                         │
                         ▼
                Collective Visual Generation
                         │
                         ▼
           Display on Stadium Screens / LED Walls
```

## Technical Approach

### 1. Audience Representation
Each audience interaction is converted into a numerical representation. Possible modalities:
- Text (Sentence Transformers)
- Emoji embeddings
- Image or doodle embeddings (CLIP)

These embeddings provide a common semantic representation of audience interactions.

### 2. Semantic Aggregation
Rather than conditioning the diffusion model directly on thousands of individual responses, audience embeddings are first aggregated into a unified representation. Potential strategies:
- Mean pooling
- Attention-based pooling
- Cluster-aware aggregation
- Weighted semantic fusion

The goal is to preserve dominant audience themes while reducing noise from individual responses.

### 3. Event-Aware Conditioning
The aggregated audience representation is combined with structured event metadata such as **artist, song, genre, venue, and event theme**, so the generated visuals reflect both audience sentiment and event context.

### 4. Conditional Diffusion
A conditional DDPM generates visuals using the unified conditioning representation. Potential experiments:
- Classifier-Free Guidance
- Different guidance scales
- DDPM vs DDIM sampling
- Sampling speed vs image quality

### 5. Dynamic Visual Evolution *(Stretch Goal)*
Rather than producing isolated images, the generated visuals can evolve throughout the performance by periodically updating the conditioning representation with new audience interactions. Instead of full video generation, smooth transitions may be achieved through **latent interpolation** between consecutive generated outputs.

## Course Alignment

| Course Topic | Application |
|--------------|-------------|
| Probabilistic Deep Generative Models | DDPM formulation and probabilistic sampling |
| GAN / WGAN | Baseline comparison for image generation |
| Variational Autoencoders | Optional latent image representation or comparison with latent diffusion approaches |
| VQ-VAE | Latent-space diffusion (if adopted) |
| DDPM | Primary generative model |
| Conditional Diffusion | Crowd-driven visual generation |
| Sampling | DDPM vs DDIM, guidance strategies |
| Quantization | Optional latent compression through VQ-VAE |

> **Note:** The project can be implemented using pixel-space diffusion initially. If time permits, a latent diffusion variant using a pretrained VQ-VAE can be explored as an extension.

## Research Questions

- **RQ1.** How can multimodal audience interactions be effectively aggregated into a unified conditioning representation?
- **RQ2.** Can conditional diffusion generate coherent visuals that reflect collective audience sentiment while maintaining visual diversity?
- **RQ3.** How do different aggregation strategies affect semantic consistency and image quality?
- **RQ4.** How does conditional diffusion compare with GAN-based approaches for crowd-driven visual generation?

## Evaluation

**Quantitative:** FID · CLIP Score · inference latency · sampling time · image diversity.

**Semantic consistency** — how well generated visuals align with the aggregated audience interactions:
- CLIP similarity
- Human evaluation
- Prompt relevance

**Qualitative** — a user study evaluating:
- Visual coherence
- Perceived emotional relevance
- Representation of audience participation
- Overall visual appeal

## Expected Contributions

- A prototype framework for **crowd-driven visual generation** in live events.
- Investigation of **many-to-one conditioning** for diffusion models.
- Comparative analysis of aggregation strategies for multimodal audience inputs.
- Evaluation of conditional diffusion models in an interactive live-event setting.

## Future Extensions

**Phase 2 — Personalized AI Memories.** Reuse the same audience and event representations to generate individualized post-event memories ("Concert Wrapped") for attendees.

**Phase 3 — Adaptive Stage Visuals.** Extend the framework to continuously generate stage visuals in response to evolving audience participation during live performances.

---

## Design Rationale

This framing was chosen deliberately to stay academically grounded and avoid overclaiming. In particular, it:

- Avoids claiming that diffusion is directly conditioned on thousands of inputs; instead it introduces a **unified conditioning representation**.
- Treats **many-to-one conditioning** as the research question rather than claiming a novel architecture.
- Does not force VAE into the pipeline unless latent diffusion is actually implemented, while leaving room to explore it in alignment with the course.
- Includes **event metadata** as an important conditioning signal, making the generated visuals specific to the concert rather than only the audience responses.
- Frames animation as **latent interpolation** rather than full video diffusion, which is more feasible for a semester project.
- Separates **image quality** from **semantic consistency** in the evaluation — important because metrics like FID alone do not indicate whether the generated visual actually reflects the crowd's collective input.
