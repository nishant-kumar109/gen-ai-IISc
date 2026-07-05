# Quiz 5 — Variational Autoencoders (VAEs)

> A multiple-choice set covering the VAE: the ELBO (reconstruction + KL terms), the
> reparameterization trick, the encoder's Gaussian outputs and the approximate posterior,
> the prior on z, why the latent space is smooth, generation after training, and how the
> encoder differs from PCA. Each question lists the options, the correct answer (marked ✅),
> and a short explanation.

---

## 1. Why Reparameterization Is Needed

**Q:** Reparameterization is needed in the formulation of the VAE because:

- a) Encoder network is not invertible
- **b) The objective function involves the computation of an expectation over the unknown latent posterior ✅**
- c) The objective function involves the computation of squared error between the real and the generated data

**Explanation:** The ELBO contains an expectation **E<sub>q(z|x)</sub>[·]** over the latent. We
need gradients w.r.t. the encoder parameters φ *through* that sampling step — and you can't
differentiate through a raw random draw. Reparameterization (z = μ + Σ<sup>½</sup>·ε,
ε∼𝒩(0,I)) moves the randomness to ε and makes the gradient computable.

---

## 2. What the KL Divergence Term Encourages

**Q:** The KL divergence term encourages:

- a) Decoder weights to become zero
- **b) Approximate posterior to match the prior ✅**
- c) Larger latent dimension
- d) Perfect reconstruction only

**Explanation:** The KL term is D<sub>KL</sub>(q(z|x) ‖ p(z)), which pulls the approximate
posterior **q(z|x)** toward the prior **p(z) = 𝒩(0, I)**.

---

## 3. The General Working Principle

**Q:** The general working principle of a deep latent variable variational model is:

- **a) obtain a lower bound on the data likelihood and optimise over the parameters of the variational distribution so that the bound becomes tighter ✅**
- b) obtain a lower bound on the joint data and the latent likelihood and optimise over the parameters of the variational distribution so that the bound becomes tighter
- c) optimise over the parameters of the data likelihood

**Explanation:** The ELBO is a lower bound on the **data** (marginal) log-likelihood
log p(x) — not the joint log p(x,z). We optimize the variational parameters to tighten it.

---

## 4. Where the Latent Variable Can Come From

**Q:** In a latent variable model, the latent variable can be from:

- a) Should be Discrete Distribution, not Continuous Distribution
- b) Should not be a Discrete Distribution, can be Continuous Distribution
- **c) can be from Discrete Distribution or Continuous Distribution ✅**
- d) can not be from Discrete Distribution or Continuous Distribution

**Explanation:** Latents may be **discrete** (e.g., a GMM / clustering / VQ-VAE) or
**continuous** (the standard VAE with a Gaussian latent).

---

## 5. Why the ELBO Is Maximized

**Q:** The Evidence Lower Bound (ELBO) is maximized because:

- a) It equals the likelihood exactly
- **b) It provides a lower bound on the log-likelihood ✅**
- c) It minimizes entropy directly
- d) It maximizes classification accuracy

**Explanation:** ELBO ≤ log p(x). Since direct likelihood maximization is intractable,
maximizing the bound pushes up the log-likelihood and tightens the gap.

---

## 6. Why VAEs Have Smooth Latent Spaces

**Q:** Which of the following best explains why VAEs generate smooth latent spaces?

- a) Because the encoder is linear
- **b) Because the KL divergence regularizes the latent distribution ✅**
- c) Because the decoder has many layers
- d) Because the optimizer is Adam

**Explanation:** The KL term forces every posterior toward the same prior 𝒩(0, I), producing a
continuous, well-packed latent space — so nearby points decode to similar data.

---

## 7. Encoder vs PCA

**Q:** Encoder reduces the dimension of data space to latent space; PCA can also reduce the
dimension. Which is a major difference between encoder and PCA?

- **a) PCA does not care about the data structure in the reduced dimension whereas encoder needs to preserve the data structure in the latent dimension ✅**
- b) Encoder does not care about the data structure in the reduced dimension whereas PCA needs to preserve the data structure in the latent dimension
- c) Both PCA and Encoder care about the data structure in reduced dimension
- d) Both PCA and Encoder do not care about the data structure in reduced dimension

**Explanation:** PCA is a *linear* projection that just maximizes variance along principal
directions. The (neural) encoder is trained so the decoder can reconstruct — i.e., it is
optimized to **preserve the data structure** in the latent space. *(Loosely worded question;
option (a) is the intended answer.)*

---

## 8. Primary Purpose of the Reparameterization Trick

**Q:** The primary purpose of the reparameterization trick is:

- a) Reduce reconstruction error
- **b) Enable gradient backpropagation through sampling ✅**
- c) Reduce model size
- d) Improve clustering

**Explanation:** It rewrites the stochastic sample as a deterministic function of the
parameters plus noise (z = μ + Σ<sup>½</sup>·ε), so gradients can flow back through the
sampling operation into the encoder.

---

## 9. What the ELBO Consists Of

**Q:** The ELBO in VAE consists of:

- a) Reconstruction term only
- b) KL divergence only
- **c) Reconstruction term and KL divergence ✅**
- d) Cross entropy only

**Explanation:** ELBO = E<sub>q(z|x)</sub>[log p(x|z)] (reconstruction) − D<sub>KL</sub>(q(z|x) ‖ p(z)) (regularization).

---

## 10. The Prior on the Latent Variable

**Q:** The prior distribution on the latent variable is commonly chosen as:

- a) Bernoulli distribution
- b) Uniform distribution
- **c) Standard Normal distribution ✅**
- d) Poisson distribution

**Explanation:** p(z) = 𝒩(0, I) — smooth, easy to sample, and reparameterizable.

---

## 11. Generation After Training

**Q:** Which of the following is true about a VAE?

- **a) Post training we can discard the encoder and just by sampling from the latent space and generate new datapoints ✅**
- b) Encoder is needed even after training to generate new samples
- c) After training we need the encoder to assist in sampling in latent space as there is no other way

**Explanation:** To generate, sample z ∼ p(z) = 𝒩(0, I) and pass it through the **decoder**.
The encoder is used only for training/inference of the posterior, not for generation.

---

## 12. What the Reconstruction Term Encourages

**Q:** The reconstruction term encourages the decoder to:

- a) Produce arbitrary outputs
- **b) Reconstruct the input accurately ✅**
- c) Maximize latent variance
- d) Minimize dataset size

**Explanation:** The term E<sub>q(z|x)</sub>[log p(x|z)] rewards the decoder for assigning high
likelihood to the original input — i.e., accurate reconstruction.

---

## 13. What the Encoder Outputs

**Q:** The encoder outputs:

- a) Only the latent vector
- **b) Mean and variance parameters of a Gaussian ✅**
- c) Class probabilities
- d) Reconstruction error

**Explanation:** The encoder outputs the parameters **μ<sub>φ</sub>(x)** and
**Σ<sub>φ</sub>(x)** of the approximate posterior q(z|x) = 𝒩(μ<sub>φ</sub>, Σ<sub>φ</sub>);
z is then sampled from it.

---

## 14. VAE as an Optimization Problem

**Q:** A VAE is:

- **a) An optimization problem where the optimal encoder and decoder are found by finding the 'arg min' of the reconstruction error between data and encoded-decoded data ✅**
- b) ... 'arg min' of the reconstruction error between data and decoded-encoded data
- c) ... 'arg max' of the reconstruction error between data and encoded-decoded data
- d) ... 'arg max' of the reconstruction error between data and decoded-encoded data

**Explanation:** Data is first **encoded then decoded** (x → z → x̂), so "encoded-decoded," and
the reconstruction error is **minimized** (arg min) — with the KL term added on top. The
traps are the order (b/d reverse it) and the direction (c/d use arg max).

---

## 15. What Parameterizes the Approximate Posterior

**Q:** Which pair of quantities parameterizes the approximate posterior?

- **a) Mean and covariance (or variance) ✅**
- b) Mean and entropy
- c) Covariance and likelihood
- d) Gradient and Hessian

**Explanation:** q(z|x) = 𝒩(μ, Σ), so it is fully specified by its **mean** and
**covariance/variance**.

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | b | 5 | b | 9  | c | 13 | b |
| 2 | b | 6 | b | 10 | c | 14 | a |
| 3 | a | 7 | a | 11 | a | 15 | a |
| 4 | c | 8 | b | 12 | b |    |   |
