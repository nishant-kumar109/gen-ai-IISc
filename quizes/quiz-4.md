# Quiz 4 — Latent Variable Models & Encoder–Decoder Architectures

> A multiple-choice set covering latent variable models: the encoder/decoder roles and the
> distributions they model (P(z|x) vs P(x|z)), the latent variable z and its prior, embeddings,
> sampling-based generation, and how GANs differ from encoder–decoder latent-variable models.
> Each question lists the options, the correct answer (marked ✅), and a short explanation.

---

## 1. The Decoder Transformation

**Q:** The decoder performs the transformation:

- a) Y → Z
- b) X → Z
- c) X → Y
- **d) Z → X ✅**

**Explanation:** The decoder takes a latent vector z and produces a data sample, so it maps
**Z → X**. (The encoder does the reverse, X → Z.)

---

## 2. A Major Application After Training

**Q:** After training a latent variable model, one major application is:

- a) Increasing dataset size
- **b) Embedding extraction ✅**
- c) Feature normalization
- d) Data compression only

**Explanation:** The learned latent z is a compact **representation/embedding** of the data,
reusable for downstream tasks. ("Data compression *only*" is too narrow — the word *only*
makes it wrong.)

---

## 3. One Latent per Datapoint

**Q:** For every datapoint x<sub>i</sub>, there exists:

- a) One optimizer
- b) One loss function
- **c) One corresponding latent variable z<sub>i</sub> ✅**
- d) One probability measure

**Explanation:** A latent variable model associates each observation x<sub>i</sub> with its own
latent code **z<sub>i</sub>**.

---

## 4. Modeling Continuous Latents

**Q:** In continuous latent variable models, latent variables are commonly modeled using:

- a) Uniform distributions only
- **b) Gaussian distributions ✅**
- c) Poisson distributions
- d) Bernoulli distributions

**Explanation:** The standard prior is a **Gaussian**, z ∼ 𝒩(0, I) — smooth, easy to sample,
and reparameterizable.

---

## 5. What the Decoder Models

**Q:** The decoder models:

- **a) P(x|z) ✅**
- b) P(z|x)
- c) P(x)
- d) P(z)

**Explanation:** Given a latent z, the decoder defines the distribution over data —
**P(x|z)** (the likelihood / data-conditional distribution).

---

## 6. GANs vs Latent-Variable Models

**Q:** Which statement correctly distinguishes GANs from latent-variable models according to
the notes?

- a) GANs estimate P(z|x) explicitly.
- **b) GANs directly generate data from latent variables, while latent-variable models employ both encoder and decoder ✅**
- c) GANs always use an encoder.
- d) GANs require discrete latent variables.

**Explanation:** A GAN's generator maps latent → data directly (no encoder/posterior),
whereas an encoder–decoder latent-variable model has **both** an encoder (X→Z) and a decoder
(Z→X).

---

## 7. What the Encoder Approximates

**Q:** The encoder approximates:

- a) P(x|z)
- **b) P(z|x) ✅**
- c) P(z)
- d) P(x)

**Explanation:** The encoder infers the latent given the data — the **posterior P(z|x)**.

---

## 8. Reducing the Latent Space to One Dimension

**Q:** "Theoretically it is not possible to use a deep neural network as an encoder and reduce
the dimension of the latent space to one." Is this statement true?

- a) Yes
- **b) No ✅**

**Explanation:** A DNN encoder can map to **any** latent dimension, including one. A 1-D
bottleneck is perfectly possible — so the claim of impossibility is **false** (answer: No).

---

## 9. What the Encoder Maps

**Q:** In an encoder–decoder architecture, the encoder primarily maps:

- a) Y → X
- b) Latent Space → X
- c) Z → X
- **d) X → Z ✅**

**Explanation:** The encoder compresses data into the latent space — **X → Z**.

---

## 10. The Quantity P(z|x)

**Q:** The quantity P(z|x) represents:

- **a) Latent posterior ✅**
- b) Prior distribution
- c) Marginal distribution
- d) Data likelihood

**Explanation:** P(z|x) is the distribution over the latent **given** the observed data — the
**posterior** over z.

---

## 11. First Step of Generation

**Q:** To generate a new datapoint using a latent variable model, one first:

- a) Computes gradients
- **b) Samples from latent variable ✅**
- c) Optimizes the encoder
- d) Computes the loss

**Explanation:** Generation = **sample z** from the prior, then pass it through the decoder.

---

## 12. Another Name for the Latent Representation

**Q:** The latent representation of a datapoint is also referred to as:

- **a) Embedding ✅**
- b) Likelihood
- c) Probability measure
- d) Gradient

**Explanation:** The latent code is the datapoint's **embedding** — its learned vector
representation.

---

## 13. What P(x|z) Is Called

**Q:** The distribution P(x|z) is called:

- a) Latent posterior
- b) Marginal distribution
- c) Prior distribution
- **d) Data conditional distribution ✅**

**Explanation:** P(x|z) is the distribution of the **data conditioned on** the latent (the
decoder's likelihood). (P(z|x) = posterior, P(x) = marginal, P(z) = prior.)

---

## 14. What the Latent Space Represents

**Q:** The latent variable space primarily represents:

- **a) Hidden representations of data ✅**
- b) Classification labels
- c) Pixel intensities
- d) Loss values

**Explanation:** The latent space holds the **hidden (compressed) representations** that
capture the data's underlying structure.

---

## 15. Producing New Samples

**Q:** New samples are produced by applying:

- a) The encoder to the sampled latent variable
- b) The loss function
- **c) The decoder to the sampled latent variable ✅**
- d) Gradient descent

**Explanation:** Sample z, then apply the **decoder** (Z → X) to turn it into a new data
sample.

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | d | 5 | a | 9  | d | 13 | d |
| 2 | b | 6 | b | 10 | a | 14 | a |
| 3 | c | 7 | b | 11 | b | 15 | c |
| 4 | b | 8 | b | 12 | a |    |   |
