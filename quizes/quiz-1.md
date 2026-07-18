# Quiz 1 — Probability Foundations for Machine Learning

> A collection of multiple-choice questions covering the probability concepts that underpin
> machine learning: probability spaces, random variables, distributions, conditional
> probability, and the i.i.d. assumption. Each question lists the options, the correct
> answer, and a short explanation.

---

## 1. Independence in Vector-Valued Random Variables

**Q:** In vector-valued random variables, independence is assumed:

- a) Between features only
- b) Between labels only
- c) Across dimensions within a datapoint
- **d) Across datapoints ✅**

**Explanation:** In ML we assume the *datapoints* are independent and identically
distributed (i.i.d.). The individual dimensions/features *within* a single datapoint are
generally correlated, so independence is **not** assumed across dimensions.

---

## 2. The Probability Triplet

**Q:** Which of the following is a valid probability triplet?

- a) A sample space, a random variable, and a measure
- **b) A sample space, event space, and a measure ✅**
- c) A sample space, event space, and a random variable
- d) Event space, a random variable, and a measure

**Explanation:** A probability space is the triplet **(Ω, ℱ, P)** — the sample space Ω,
the event space (a σ-algebra ℱ), and a probability measure P.

---

## 3. Properties of a Probability Measure

**Q:** Which of the following is always true for a probability measure?

- a) P(Ω) = 0
- b) P(A) < 0
- **c) P(A) ≥ 0 ✅**
- d) P(∅) = 1

**Explanation:** Probabilities are never negative, so P(A) ≥ 0 always holds. By contrast
P(Ω) must equal **1** (not 0), and P(∅) must equal **0** (not 1).

---

## 4. Discrete vs. Continuous Random Variables

**Q:** What is the primary distinction between a discrete and a continuous random variable?

- a) Continuous random variables are always uniformly distributed, while discrete ones are not.
- b) Discrete random variables follow a PMF, while continuous ones follow a CDF.
- c) Discrete random variables can take only integer values, while continuous ones take real numbers.
- **d) Discrete random variables are countable, while continuous random variables have an uncountable range. ✅**

**Explanation:** The defining difference is cardinality of the range: **countable**
(discrete) vs. **uncountable** (continuous). Discrete variables need not be integers, and
both discrete and continuous variables have CDFs.

---

## 5. Is It a Valid Probability Measure?

**Q:** A die is rolled with P(1)=0.3, P(2)=0.1, P(3)=0.1, P(4)=0.1, P(5)=0.1, P(6)=0.2.
Is this a proper probability measure?

- **a) Yes ✅**
- b) No
- c) Maybe

**Explanation:** All probabilities are ≥ 0, and they sum to
0.3 + 0.1 + 0.1 + 0.1 + 0.1 + 0.2 = **1.0**. Both axioms hold, so it is valid.

---

## 6. Estimating P(Y | X)

**Q:** Estimating P(Y | X) is commonly associated with:

- a) Conditional Generation
- b) Random sampling
- **c) Classification or regression ✅**
- d) Clustering only

**Explanation:** Predicting a label/target Y given an input X is the core of supervised
learning — classification (discrete Y) or regression (continuous Y).

---

## 7. The Meaning of "i.i.d."

**Q:** The notation "i.i.d." stands for:

- **a) Independent and identically distributed ✅**
- b) Inverse identical decomposition
- c) Integrated independent density
- d) Iterative information distribution

**Explanation:** Samples are assumed to be drawn independently from the *same* underlying
distribution — a foundational assumption for most learning theory.

---

## 8. Conditional Probability

**Q:** Conditional probability P(A | B) is given by:

- a) P(A) / P(B)
- **b) P(A ∩ B) / P(B) ✅**
- c) P(A ∪ B)
- d) P(A)·P(B)

**Explanation:** P(A | B) = P(A ∩ B) / P(B), defined for P(B) > 0. Note: P(A)·P(B) equals
P(A ∩ B) **only** when A and B are independent.

---

## 9. The Sample Space

**Q:** The sample space is:

- a) The set of impossible events
- b) The optimization domain
- **c) The set of all possible outcomes ✅**
- d) A collection of labels

**Explanation:** The sample space Ω is the set of *all* possible outcomes of a random
experiment.

---

## 10. Probability Density Functions

**Q:** Which statement about probability density functions is correct?

- a) Every random variable has a density function
- b) Density values are probabilities themselves
- c) Density functions are always negative
- **d) Density values do not directly correspond to probabilities ✅**

**Explanation:** A PDF value can exceed 1 and is **not** a probability by itself. Only the
*integral* of the density over an interval yields a probability. (Also, not every random
variable admits a density — e.g. discrete ones.)

---

## 11. Range Space of Vector-Valued Random Variables

**Q:** In machine learning, vector-valued random variables usually have range space:

- a) F
- **b) ℝᵈ ✅**
- c) ℝ
- d) ℤ

**Explanation:** A vector-valued random variable maps to a *d*-dimensional real vector, so
its range is **ℝᵈ**. Plain ℝ is the range for scalar (single-valued) random variables.

---

## 12. Obtaining a Marginal Distribution

**Q:** Marginal distribution can be obtained from a joint distribution by:

- a) Optimization
- b) Matrix inversion
- **c) Integration over the other variable ✅**
- d) Differentiation

**Explanation:** Marginalization means "summing out" (discrete) or "integrating out"
(continuous) the other variable(s): p(x) = ∫ p(x, y) dy.

---

## 13. Training a Machine Learning Model

**Q:** The training of a machine learning model can be formulated as:

- **a) Solving an optimization problem over parameters ✅**
- b) Removing probability distributions
- c) Maximizing sample space
- d) Estimating only labels

**Explanation:** Training searches for parameters θ that minimize a loss (or maximize a
likelihood) — fundamentally an optimization problem.

---

## 14. Probability of a Union of Disjoint Events

**Q:** If A ∩ B = ∅, then P(A ∪ B) = ?

- **a) P(A) + P(B) ✅**
- b) 0
- c) P(A) − P(B)
- d) P(A)·P(B)

**Explanation:** When A and B are disjoint (mutually exclusive), additivity gives
P(A ∪ B) = P(A) + P(B), since the overlap term P(A ∩ B) = 0.

---

## 15. Formal Definition of a Random Variable

**Q:** A random variable is formally defined as:

- a) X : Ω → F
- b) X : ℝ → Ω
- c) X : F → Ω
- **d) X : Ω → ℝ ✅**

**Explanation:** A random variable is a (measurable) function from the sample space to the
real numbers: **X : Ω → ℝ**.

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|
| 1 | d | 6 | c | 11 | b |
| 2 | b | 7 | a | 12 | c |
| 3 | c | 8 | b | 13 | a |
| 4 | d | 9 | c | 14 | a |
| 5 | a | 10 | d | 15 | d |
