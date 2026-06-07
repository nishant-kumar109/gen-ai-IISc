# Quiz 2 — Probability, Random Variables & Learning Objectives in ML

> A multiple-choice set covering sample spaces, random variables, conditional vs.
> unconditional generation, supervised learning objectives, MLE, and how images/data are
> represented as vectors. Each question lists the options, the correct answer (marked ✅),
> and a short explanation.

---

## 1. Example of a Sample Space

**Q:** Which of the following is an example of a sample space?

- a) ℝᵈ
- b) Pₓ
- **c) {H, T} ✅**
- d) {0, 1}ᵈ

**Explanation:** A sample space is the *set of all possible outcomes* of an experiment.
{H, T} is the outcome set of a coin flip. ℝᵈ and {0,1}ᵈ are range spaces, and Pₓ is a
probability distribution — not sample spaces.

---

## 2. Unconditional Generation

**Q:** Unconditional generation attempts to model:

- **a) Pₓ ✅**
- b) P_{Y|X}
- c) P_{X|Y}
- d) P_Y

**Explanation:** Unconditional generation models the data distribution **P(X)** directly —
no conditioning variable is involved.

---

## 3. Purpose of Probability Theory in ML

**Q:** The purpose of introducing probability theory in machine learning is to handle:

- a) Computational complexity
- b) Hardware limitations
- c) Optimization only
- **d) Uncertainty arising from observations ✅**

**Explanation:** Probability gives us a principled framework for reasoning about the
**uncertainty** inherent in noisy, incomplete observations.

---

## 4. What a Datapoint Is an Element Of

**Q:** Every datapoint in a dataset is an element of:

- a) The event space
- b) The parameter space
- **c) The sample space ✅**
- d) The range space of a random variable

**Explanation:** Each datapoint is an outcome of the data-generating process, so it lives in
the **sample space** Ω.

---

## 5. Conditional Generation

**Q:** Conditional generation primarily models:

- a) P_Y
- **b) P_{X|Y} ✅**
- c) P_{XY}
- d) Pₓ

**Explanation:** Conditional generation produces data X *given* a condition/label Y — i.e.
it models **P(X | Y)** (e.g. text-to-image generation).

---

## 6. Bounding-Box Prediction

**Q:** Bounding-box prediction is typically treated as:

- a) Clustering
- b) Classification
- c) Density estimation
- **d) Regression ✅**

**Explanation:** A bounding box is described by continuous coordinates (x, y, width,
height), so predicting it is a **regression** problem.

---

## 7. Semantic Segmentation

**Q:** Semantic segmentation can be viewed as learning:

- a) P_Y
- b) P_{X|Y}
- c) Pₓ
- **d) P_{Y|X} ✅**

**Explanation:** Segmentation assigns a label to every pixel given the image — per-pixel
classification, i.e. modeling **P(Y | X)**.

---

## 8. Neural Network Approximating a Distribution

**Q:** A neural network used to approximate a distribution is an example of:

- **a) A parametric functional form ✅**
- b) A random variable
- c) A probability measure
- d) A sample space

**Explanation:** A neural net has learnable parameters θ and defines a family of functions —
it is a **parametric functional form** used to approximate a distribution.

---

## 9. Raw Speech Signal in ASR

**Q:** In Automatic Speech Recognition (ASR), the raw speech signal is first converted into:

- a) Phonemes
- b) Labels
- **c) Voltages measured by a microphone ✅**
- d) Words

**Explanation:** Sound is captured physically by a microphone as **voltage measurements**;
phonemes and words come later in the pipeline.

---

## 10. Name

**Answer:** Nishant Kumar

---

## 11. Supervised Learning Objective

**Q:** Given a dataset D = {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, the primary supervised learning objective is to
estimate:

- a) Pₓ
- b) P_{X|Y}
- **c) P_{Y|X} ✅**
- d) P_Y

**Explanation:** Supervised learning predicts the label given the input, i.e. it estimates
**P(Y | X)**.

---

## 12. Model vs. True Distribution After Training

**Q:** After training successfully, the desired relationship between the model and true
distribution is:

- **a) P_θ ≈ Pₓ ✅**
- b) P_θ ⊥ Pₓ
- c) P_θ = P_Y
- d) P_θ = 0

**Explanation:** A well-trained model's distribution P_θ should **approximate** the true
data distribution Pₓ.

---

## 13. Maximum Likelihood Estimation

**Q:** Maximum Likelihood Estimation (MLE) is obtained by:

- **a) Maximizing likelihood under the model ✅**
- b) Maximizing variance
- c) Minimizing dataset size
- d) Minimizing label entropy

**Explanation:** MLE chooses the parameters θ that **maximize the likelihood** of the
observed data under the model.

---

## 14. General Principle of Generative Models

**Q:** The general principle followed in generative models is:

- a) Parameterize the true data distribution with a neural network and optimise via
  **maximisation** of a divergence metric between the true and the parametric **conditional**
  distributions
- **b) Parameterize the true data distribution with a neural network and optimise via
  *minimisation* of a divergence metric between the true and the parametric distributions ✅**
- c) Use a parametric density model and estimate the parameters via label-based supervised loss
- d) All of the above (Not considering None of the Above)
- e) None of the Above

**Explanation:** Generative models fit a parametric distribution P_θ and train it by
**minimising a divergence** (e.g. KL divergence) between the true distribution and P_θ.
Option (a) is wrong on two counts ("maximisation" and "conditional").

---

## 15. Why Random Variables Are Introduced

**Q:** Why are random variables introduced?

- a) To perform optimization
- b) To remove uncertainty
- **c) To convert inaccessible outcomes into measurable numerical quantities ✅**
- d) To increase dimensionality

**Explanation:** A random variable maps abstract outcomes in Ω to **numbers**, making them
measurable and easy to compute with. (They don't *remove* uncertainty — they let us quantify
it.)

---

## 16. Vector Representation of an Image

**Q:** For an image of size 300×200×3, a datapoint can naturally be represented as a vector
in:

- a) ℝ⁶⁰⁰⁰⁰
- **b) ℝ¹⁸⁰⁰⁰⁰ ✅**
- c) ℝ³⁰⁰
- d) ℝ³

**Explanation:** Flattening the image gives 300 × 200 × 3 = **180,000** values, so it lives
in **ℝ¹⁸⁰⁰⁰⁰**.

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | c | 5 | b | 9 | c | 13 | a |
| 2 | a | 6 | d | 10 | (name) | 14 | b |
| 3 | d | 7 | d | 11 | c | 15 | c |
| 4 | c | 8 | a | 12 | a | 16 | b |
