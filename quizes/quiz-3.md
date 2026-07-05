# Quiz 3 — Vanilla GANs (Generative Adversarial Networks)

> A multiple-choice set covering the Vanilla GAN setup: the generator/discriminator roles,
> the latent variable z, the min-max adversarial objective, the discriminator loss, training
> dynamics, and notation (g_θ, D_w). Each question lists the options, the correct answer
> (marked ✅), and a short explanation.

---

## 1. Correct Statement About Vanilla GANs

**Q:** Which statement is correct about Vanilla GANs?

- a) The discriminator maps labels to latent vectors
- b) The generator directly receives real images as input
- c) The discriminator directly generates fake images
- **d) The generator maps latent noise to generated samples ✅**

**Explanation:** The generator G takes a latent noise vector z and maps it to a sample,
G(z). It never sees real images directly, and the discriminator only *scores* samples — it
doesn't generate them.

---

## 2. Role of the Generator

**Q:** In a Vanilla GAN, what is the main role of the generator network?

- **a) To generate samples that look like they came from the real data distribution ✅**
- b) To reduce the dimension of the input data
- c) To compute the exact probability density of real data
- d) To classify real samples into different classes

**Explanation:** The generator's job is to produce samples whose distribution matches the
real data distribution Pₓ — fooling the discriminator. It never computes an explicit density.

---

## 3. Role of the Discriminator

**Q:** In a Vanilla GAN, what is the role of the discriminator?

- **a) To distinguish real samples from generated samples ✅**
- b) To generate fake samples from noise
- c) To reconstruct input samples
- d) To estimate the latent variable directly

**Explanation:** The discriminator D is a binary classifier that outputs the probability a
sample is **real** vs. **generated**.

---

## 4. Generator Becomes Very Good

**Q:** If the generator becomes very good, what should happen to the discriminator's ability?

- a) It should easily distinguish real and fake samples
- b) It should output only zero
- c) It should stop receiving real samples
- **d) It should become confused between real and generated samples ✅**

**Explanation:** At the ideal equilibrium the generated and real distributions match, so the
discriminator cannot tell them apart and outputs D(x) ≈ 0.5 everywhere.

---

## 5. Meaning of the Latent Variable z

**Q:** In Vanilla GAN notation, if z ~ 𝒩(0, I), what does z represent?

- a) A class label
- b) The output of the discriminator
- c) A real data sample
- **d) A random latent input to the generator ✅**

**Explanation:** z is sampled from a simple prior (standard Gaussian) and fed into the
generator as its **random latent input**; the generator transforms it into a sample.

---

## 6. The Term log D_w(x) in the Discriminator Loss

**Q:** In the discriminator loss, the term log D_w(x) encourages:

- **a) Real samples to get discriminator output close to 1 ✅**
- b) Latent vectors to become Gaussian
- c) Generated samples to get discriminator output close to 1
- d) Real samples to get discriminator output close to 0

**Explanation:** Here x is a **real** sample (x ~ Pₓ). Maximizing log D_w(x) pushes D(x) → 1,
i.e. the discriminator learns to label real data as real.

---

## 7. Discriminator Output Range

**Q:** What is the usual output range of the discriminator in a Vanilla GAN?

- a) [-1, 1]
- b) (-∞, +∞)
- c) {1, 2, 3, …, k} : k is number of classes
- **d) [0, 1] ✅**

**Explanation:** The discriminator outputs a **probability** (real vs. fake), so its range is
[0, 1] — typically produced by a sigmoid.

---

## 8. Nature of the Optimization Problem

**Q:** The Vanilla GAN optimization problem is best described as:

- **a) A min-max adversarial optimization problem ✅**
- b) A supervised classification problem only
- c) A clustering problem
- d) A reconstruction loss minimization problem

**Explanation:** GANs solve min_θ max_w J(θ, w) — the generator and discriminator play an
**adversarial min-max game**.

---

## 9. Discriminator Too Strong Early in Training

**Q:** What happens if the discriminator is too strong early in training?

- a) The discriminator stops being a classifier
- b) The generator immediately becomes perfect
- c) The latent distribution becomes equal to the data distribution
- **d) The generator may receive poor or weak learning signal ✅**

**Explanation:** If D is near-perfect, it confidently rejects all fakes and the generator's
gradient vanishes — it gets little useful signal to improve (the vanishing-gradient problem).

---

## 10. What the Discriminator Maximizes

> *Enter the option in small case without any spaces.*

**Q:** In the Vanilla GAN objective, the discriminator tries to maximize:

- **a) E_{x~pₓ}[log D_w(x)] + E_{x̂~p_θ}[log(1 − D_w(x̂))] ✅**
- b) E_z[log g_θ(z)]
- c) E_{x~pₓ}[log(1 − D_w(x))]
- d) E_{x~pₓ}[D_w(x)²]

**Answer to enter:** `a`

**Explanation:** The discriminator maximizes the full objective: push D → 1 on **real** data
(first term) and D → 0 on **generated** data (second term). Option (c) is only the fake-sample
half; (b) and (d) are not the GAN loss.

---

## 11. The Practical Discriminator Update

**Q:** Which of the following correctly describes the practical discriminator update?

- a) Use only real data and update both networks
- b) Use only generated data and freeze both networks
- **c) Sample real data and generated data, pass both through discriminator, update only w ✅**
- d) Sample only latent vectors and update only θ

**Explanation:** A discriminator step uses **both** real and generated samples and updates
**only the discriminator parameters w**, keeping the generator fixed.

---

## 12. Which Parameters Are Fixed When Updating the Discriminator

**Q:** In GAN training, when updating the discriminator, which parameters are usually kept
fixed?

- a) Discriminator parameters
- **b) Generator parameters ✅**
- c) Both of Generator and Discriminator

**Explanation:** Training alternates: when updating D you **freeze the generator (θ)** and
adjust only w; when updating G you freeze D.

---

## 13. Meaning of min_θ max_w J(θ, w)

**Q:** The Vanilla GAN objective can be written as min_θ max_w J(θ, w). What does this mean?

- **a) The generator minimizes and the discriminator maximizes the adversarial objective ✅**
- b) The discriminator minimizes while the generator maximizes
- c) The generator and discriminator are trained independently
- d) Both generator and discriminator minimize the same objective

**Explanation:** θ (generator) sits under **min** and w (discriminator) under **max** — the
generator minimizes the objective while the discriminator maximizes it.

---

## 14. Meaning of θ in g_θ(z)

**Q:** If the generator is denoted by g_θ(z), what does θ represent?

- a) The discriminator output
- **b) Parameters of the generator ✅**
- c) Parameters of the discriminator
- d) The real data distribution

**Explanation:** The subscript θ denotes the **learnable parameters (weights) of the
generator** network. (By the same convention, w in D_w denotes the discriminator's parameters.)

---

## 15. Interpretation of D_w(x) = 1

**Q:** If D_w(x) = 1, the discriminator believes that x is:

- a) A class label
- b) Definitely fake
- c) A latent vector
- **d) Definitely real ✅**

**Explanation:** The discriminator output is the probability that x is real, so D_w(x) = 1
means "**definitely real**" (and 0 means definitely fake).

---

## 16. Classifier-Guided Interpretation of GANs

**Q:** In the classifier-guided interpretation of GANs, what does the discriminator help the
generator learn?

- a) The number of classes in the dataset
- b) The learning rate schedule
- **c) Whether generated samples are distinguishable from real samples ✅**
- d) The exact inverse of the generator

**Explanation:** The discriminator acts as a learned critic: its signal tells the generator
**how distinguishable its fakes still are from real data**, guiding it to make them more
realistic.

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | d | 5 | d | 9  | d | 13 | a |
| 2 | a | 6 | a | 10 | a | 14 | b |
| 3 | a | 7 | d | 11 | c | 15 | d |
| 4 | d | 8 | a | 12 | b | 16 | c |
