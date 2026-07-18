# Quiz 8 — Denoising Diffusion Probabilistic Models (DDPM)

> A multiple-choice set covering the DDPM formulation: the fixed forward (noising) process and
> the learned reverse (denoising) process, the reverse-Gaussian parameters, the role of Bayes'
> rule and the ELBO objective, the connection to hierarchical VAEs, and the stationary noise
> distribution. Each question lists the options, the correct answer (marked ✅), and a short
> explanation.

---

## 1. Forward vs Reverse Parameters

**Q:** Choose the correct answer.

- a) Parameters of Forward process is learnt and Reverse diffusion process is deterministic
- b) Parameters of Forward and Reverse diffusion process are learnt
- c) Parameters of Forward and Reverse diffusion process are deterministic
- **d) Parameters of Forward is deterministic and Reverse diffusion process is learnt ✅**

**Explanation:** The forward process uses a **fixed** (predefined) noise schedule — no learned
parameters — while the **reverse** denoising process is parameterised by a network and
**learned**.

---

## 2. DDPM Encoder vs VAE Encoder

**Q:** In DDPM, the encoder differs from a VAE because:

- a) It is jointly optimized with the decoder.
- **b) It is deterministic and fixed ✅**
- c) It is deeper.
- d) It uses transformers.

**Explanation:** The DDPM "encoder" is the forward noising process — **fixed and non-learnable**
— unlike a VAE encoder, which is learned jointly with the decoder.

---

## 3. What T Denotes

**Q:** In the DDPM formulation, T denotes:

- **a) Number of diffusion steps ✅**
- b) Training epochs
- c) Number of classes
- d) Batch size

**Explanation:** T is the total number of forward/reverse **diffusion timesteps** (x₀ → x_T).

---

## 4. Reverse-Process Gaussian Parameters

**Q:** The reverse-process Gaussian parameters are functions of:

- a) x_1
- b) x_0
- c) x_(t-1)
- **d) x_t ✅**

**Explanation:** The reverse step P_θ(x_{t-1} | x_t) = 𝒩(x_{t-1}; μ_θ(x_t), Σ_θ(x_t)) — its mean
and covariance are functions of the **current** noisy sample x_t.

---

## 5. Role of Bayes' Rule in the ELBO

**Q:** Bayes' rule is primarily used during the ELBO derivation to:

- **a) Reverse conditional probabilities ✅**
- b) Normalize images
- c) Compute gradients
- d) Remove Gaussian noise

**Explanation:** Bayes' rule rewrites the forward transition into the **tractable reverse
posterior** q(x_{t-1} | x_t, x₀), which the ELBO's KL terms then match.

---

## 6. Learned Parameters of the Reverse Gaussian

**Q:** The parameters learned in the reverse Gaussian distribution are:

- a) Mean only
- b) Variance only
- c) Covariance only
- **d) Mean and variance ✅**

**Explanation:** The reverse Gaussian is defined by its **mean and variance** (μ_θ, Σ_θ). *(Note:
the original DDPM of Ho et al. learns only the mean and **fixes** the variance; per the course
formulation both are learnable, so **d**.)*

---

## 7. Final Training Objective — What Is Learned

**Q:** The final objective of training a DDPM is to learn:

- a) The input data labels
- b) The covariance of the dataset
- c) The forward diffusion schedule
- **d) The reverse denoising process ✅**

**Explanation:** Training fits the parameters of the **reverse (denoising) process** so that
running it from noise reconstructs the data distribution.

---

## 8. Purpose of the Decoder

**Q:** The primary purpose of the decoder in DDPM is to:

- a) Learn latent embeddings
- b) Compute posterior distributions
- c) Add Gaussian noise
- **d) Generate data samples ✅**

**Explanation:** The decoder = the reverse process; walking it from x_T ~ 𝒩(0, I) back to x₀
**generates** new samples.

---

## 9. How Forward Latents Are Obtained

**Q:** During the forward process, each latent variable is obtained by:

- a) Averaging previous latent variables
- b) Multiplying by random matrices
- c) Removing Gaussian noise
- **d) Adding Gaussian noise ✅**

**Explanation:** x_t = √(α_t)·x_{t-1} + √(1−α_t)·ε — each step **adds** Gaussian noise to the
previous latent.

---

## 10. What the Reverse Process Estimates

**Q:** The reverse process estimates:

- a) P(X_t | X_{t-1})
- b) Q(X_t)
- **c) P(X_{t-1} | X_t) ✅**
- d) P(X_0)

**Explanation:** The learned reverse transition denoises one step: **P(x_{t-1} | x_t)**.

---

## 11. Hierarchical VAE vs Standard VAE

**Q:** In a hierarchical VAE, compared to a standard VAE:

- a) The encoder is deterministic.
- b) There is only one latent variable.
- **c) There are multiple latent variables arranged hierarchically ✅**
- d) The decoder is removed.

**Explanation:** A hierarchical VAE stacks **multiple latent variables** in a hierarchy — the
structure DDPM mirrors with its chain of latents x_1 … x_T.

---

## 12. Stationary Distribution of the Forward Chain

**Q:** The stationary distribution of the forward Markov chain is approximately:

- a) Laplace
- **b) 𝒩(0, I) ✅**
- c) Beta
- d) Uniform

**Explanation:** As t → T the forward process drives any input toward an isotropic standard
Gaussian, **𝒩(0, I)**.

---

## 13. DDPMs as a Special Case

**Q:** DDPMs can be viewed as a special case of:

- a) GANs
- **b) Hierarchical VAEs ✅**
- c) Normalizing Flows
- d) Autoencoders

**Explanation:** A DDPM is a **hierarchical (Markovian) VAE** with a fixed encoder and a chain
of latents of the same dimensionality as the data.

---

## 14. Training Objective

**Q:** The objective of training a DDPM is to optimize:

- a) Hinge Loss
- b) Cross-entropy
- c) Mean Absolute Error
- **d) ELBO ✅**

**Explanation:** Like a VAE, a DDPM maximises the **Evidence Lower Bound** on the data
log-likelihood (which reduces to a denoising/noise-prediction loss).

---

## 15. Aim of Forward Diffusion

**Q:** The aim of forward diffusion is to convert:

- **a) Image to Isotropic Gaussian Noise ✅**
- b) Image to Non Isotropic Gaussian Noise
- c) Isotropic Gaussian Noise to Image
- d) Non Isotropic Gaussian Noise to Image

**Explanation:** The forward process gradually corrupts a data **image into isotropic Gaussian
noise**; the reverse process does the opposite (noise → image).

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | d | 5 | a | 9  | d | 13 | b |
| 2 | b | 6 | d | 10 | c | 14 | d |
| 3 | a | 7 | d | 11 | c | 15 | a |
| 4 | d | 8 | d | 12 | b |    |   |
