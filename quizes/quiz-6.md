# Quiz 6 — VAEs, VQ-VAE & Model Generalization

> A multiple-choice set covering the VAE objective (ELBO = reconstruction + KL), posterior
> collapse, VQ-VAE and discrete latents, plus the generalization toolkit that surrounds
> training: hyperparameters, regularization (incl. Batch/Layer Norm), the bias–variance
> decomposition, validation / early stopping / test-leakage, and the Law of Large Numbers
> behind sample-based MLE. Each question lists the options, the correct answer (marked ✅),
> and a short explanation.

---

## 1. What the KL Term Measures

**Q:** The KL divergence term in the ELBO measures the difference between:

- a) The data distribution and the model distribution
- **b) The latent posterior and the latent prior ✅**
- c) The training and validation distributions
- d) The encoder and decoder outputs

**Explanation:** The KL term is D<sub>KL</sub>(q(z|x) ‖ p(z)) — it pulls the approximate
**posterior** q(z|x) toward the **prior** p(z) = 𝒩(0, I).

---

## 2. Why Posterior Collapse Is Undesirable

**Q:** Posterior collapse is undesirable because:

- **a) The latent representation carries little or no useful information ✅**
- b) The decoder becomes computationally expensive
- c) The latent dimension increases
- d) Training becomes deterministic

**Explanation:** When the posterior collapses to the prior, z becomes independent of x, so the
latent code stops encoding anything useful about the input.

---

## 3. Which Is a Hyperparameter

**Q:** Which of the following is considered a hyperparameter?

- a) Weight of a neuron
- b) Bias of a neuron
- **c) Learning rate ✅**
- d) Gradient value

**Explanation:** Weights and biases are **learned parameters**; the gradient is *computed*. The
**learning rate** is set by the practitioner before/around training — a hyperparameter.

---

## 4. Batch / Layer Normalization

**Q:** Batch Normalization and Layer Normalization are examples of:

- a) Optimization algorithms
- **b) Regularization methods ✅**
- c) Loss functions
- d) Sampling methods

**Explanation:** They are normalization techniques that stabilize training and improve
generalization; among the given options they fall under **regularization methods** (not
optimizers, losses, or samplers).

---

## 5. Motivation for VQ-VAE

**Q:** The motivation behind introducing Vector Quantized VAE (VQ-VAE) is primarily to:

- a) Increase reconstruction loss
- **b) Address posterior collapse ✅**
- c) Remove the decoder
- d) Eliminate latent variables

**Explanation:** VQ-VAE learns a **discrete** latent codebook, which sidesteps the **posterior
collapse** that afflicts continuous VAEs with powerful decoders.

---

## 6. Purpose of the Validation Set

**Q:** During training, the validation dataset is primarily used to:

- a) Update model parameters
- b) Compute gradients
- **c) Tune hyperparameters and decide when to stop training ✅**
- d) Increase the size of the training data

**Explanation:** Parameters/gradients come from the **training** set; the **validation** set is
held out to tune hyperparameters and trigger early stopping.

---

## 7. What Posterior Collapse Is

**Q:** Posterior collapse in a VAE refers to the situation where:

- **a) The decoder ignores the latent variables ✅**
- b) The encoder becomes deeper
- c) The latent dimension becomes very large
- d) The reconstruction error becomes zero

**Explanation:** In collapse, the decoder reconstructs without relying on z (posterior ≈ prior),
so the latents are effectively **ignored**.

---

## 8. Bias in the Bias–Variance Decomposition

**Q:** In the bias-variance decomposition, the bias primarily measures:

- a) Sensitivity of the model to different datasets
- b) Computational complexity
- **c) How close the model is to the true distribution ✅**
- d) Number of trainable parameters

**Explanation:** **Bias** is the systematic error — how far the model's average prediction is
from the truth. (Sensitivity to different datasets is **variance**.)

---

## 9. The Latent Prior in a Standard VAE

**Q:** In a standard VAE, the latent prior is usually assumed to be:

- a) Uniform
- b) Bernoulli
- **c) Standard Gaussian ✅**
- d) Poisson

**Explanation:** p(z) = 𝒩(0, I) — smooth, easy to sample, and reparameterizable.

---

## 10. Motivation for Regularization

**Q:** Which statement best summarizes the motivation for regularization?

- **a) Regularization aims to reduce model complexity and improve generalization by increasing model bias appropriately ✅**
- b) Regularization minimizes the reconstruction loss to zero
- c) Regularization eliminates the need for validation data
- d) Regularization guarantees perfect training accuracy

**Explanation:** Regularization trades a little extra **bias** for a large reduction in
**variance**, curbing overfitting and improving **generalization**.

---

## 11. What Early Stopping Selects

**Q:** In early stopping, the model that is finally selected is typically the one with:

- a) Highest training accuracy
- b) Lowest training loss
- **c) Best validation performance ✅**
- d) Largest number of epochs

**Explanation:** Early stopping keeps the checkpoint with the best **validation** score, before
validation error starts rising (the onset of overfitting).

---

## 12. Test Data Leakage

**Q:** Test data leakage occurs when:

- a) Test data is larger than the training data
- b) Test data is accidentally deleted
- **c) Information from the test set influences model design or parameter selection ✅**
- d) Validation accuracy is higher than training accuracy

**Explanation:** Leakage = the test set (directly or indirectly) informs modeling choices, which
inflates reported performance and breaks the honesty of the final evaluation.

---

## 13. What the ELBO Consists Of

**Q:** The ELBO optimized in a Variational Autoencoder consists primarily of:

- a) Cross entropy loss and Mean Squared Error
- **b) Reconstruction likelihood and KL divergence ✅**
- c) KL divergence and Classification loss
- d) Wasserstein distance and Reconstruction loss

**Explanation:** ELBO = E<sub>q(z|x)</sub>[log p(x|z)] (reconstruction likelihood) −
D<sub>KL</sub>(q(z|x) ‖ p(z)) (regularization).

---

## 14. VQ-VAE's Latent Representation

**Q:** In a VQ-VAE, the latent representation is mapped to:

- a) A continuous Gaussian distribution
- **b) A learned discrete codebook ✅**
- c) Random noise
- d) Training labels

**Explanation:** VQ-VAE quantizes the encoder output to the nearest vector in a **learned
discrete codebook**, giving a discrete latent space.

---

## 15. Why Sample-Based MLE Works

**Q:** The sample-based approximation of Maximum Likelihood Estimation is possible because of:

- a) Bayes' Rule
- b) Central Limit Theorem
- **c) Weak Law of Large Numbers ✅**
- d) Jensen's Inequality

**Explanation:** Replacing the expected log-likelihood with a **sample average** is justified by
the **(Weak) Law of Large Numbers** (sample mean → expectation). *(Jensen's inequality justifies
the ELBO; the CLT describes the distribution of the mean, not the approximation itself.)*

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | b | 5 | b | 9  | c | 13 | b |
| 2 | a | 6 | c | 10 | a | 14 | b |
| 3 | c | 7 | a | 11 | c | 15 | c |
| 4 | b | 8 | c | 12 | c |    |   |
