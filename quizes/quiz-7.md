# Quiz 7 — Autoregressive Models, Attention & Transformers

> A multiple-choice set covering autoregressive sequence modelling, self-attention and scaled
> dot-product attention, the Query/Key/Value projections, encoder–decoder attention, RNNs and
> the vanishing-gradient problem, and how Transformers differ from RNNs. Each question lists the
> options, the correct answer (marked ✅), and a short explanation.

---

## 1. Why Autoregressive Models Suit Sequential Data

**Q:** Which statement best explains why autoregressive models naturally model sequential data?

- a) They always require recurrent networks.
- **b) Every prediction depends on previously observed variables ✅**
- c) They assume all observations are independent.
- d) Every prediction depends only on the future.

**Explanation:** Autoregression factorises the joint as p(x) = ∏ p(x_t | x_{<t}) — each variable
is predicted from the **past** ones, which is exactly the structure of sequential data.

---

## 2. The Output of One Query Vector

**Q:** The output representation corresponding to one query vector is:

- a) The average of all embeddings.
- **b) A weighted linear combination of all value vectors ✅**
- c) A weighted sum of key vectors.
- d) The query itself.

**Explanation:** Attention weights (from query–key similarity) are applied to the **value**
vectors: output = Σ softmax(qKᵀ)·V.

---

## 3. Contextual Embeddings from Self-Attention

**Q:** Which statement is TRUE regarding contextual embeddings produced by self-attention?

- **a) Every output token potentially depends on every token in the sequence ✅**
- b) Every token is processed independently.
- c) Every output token depends only on itself.
- d) Context is ignored.

**Explanation:** Self-attention lets each position attend to all positions, so every output
embedding is a function of the **whole** sequence (subject to any masking).

---

## 4. Attention in Encoder–Decoder Models

**Q:** Which statement best characterizes attention in encoder-decoder models?

- a) It removes the decoder.
- b) It predicts future words directly.
- **c) It computes a learnable weighted combination of encoder hidden states ✅**
- d) It replaces hidden states.

**Explanation:** Cross-attention lets the decoder form a context vector as a learned weighted
sum over **encoder** hidden states, instead of relying on a single final state.

---

## 5. Raster Scanning an Image for a Transformer

**Q:** Which of the following is NOT a consequence of representing an image using raster
scanning before applying a Transformer?

- a) The model can process images similarly to text.
- **b) Spatial locality is explicitly preserved by raster scanning ✅**
- c) Attention can operate over image tokens.
- d) The image becomes a sequence.

**Explanation:** Raster scanning **flattens** a 2-D image into a 1-D sequence, so 2-D spatial
locality is **lost**, not preserved — that false statement is the correct "NOT" answer.

---

## 6. Limitation of the MLP-based Neural Language Model

**Q:** What is the principal limitation of the original MLP-based Neural Language Model proposed
before recurrent networks?

- a) It cannot approximate nonlinear functions.
- **b) It ignores the ordering of words ✅**
- c) It requires attention.
- d) It cannot model probabilities.

**Explanation:** By elimination (a, c, d are false). *(Strictly, Bengio's MLP-LM concatenates
words in order within a **fixed context window** — its real limitation is that fixed window —
but among the given options, **b** is the intended answer.)*

---

## 7. Nonlinearity in Attention Scores

**Q:** Which operation introduces nonlinearity into the computation of attention scores?

- a) Linear projection
- **b) Softmax ✅**
- c) Matrix multiplication
- d) Dot product

**Explanation:** Projections, matrix multiply and dot products are all linear; the **softmax**
that normalises the scores is the nonlinearity.

---

## 8. Obtaining Query, Key and Value

**Q:** In self-attention, the Query, Key and Value matrices are obtained by:

- a) Three nonlinear neural networks.
- **b) Three learned linear projections of the input ✅**
- c) Random initialization only.
- d) Three convolutional layers.

**Explanation:** Q = XWᵠ, K = XWᵏ, V = XWᵛ — three **learned linear** projections of the same
input.

---

## 9. Dividing by √d_k

**Q:** Why is the attention score divided by √d_k before applying softmax?

- a) To reduce sequence length.
- b) To reduce computational complexity.
- c) To increase attention sparsity.
- **d) To avoid excessively large dot products that lead to unstable softmax outputs ✅**

**Explanation:** Dot products grow with dimension d_k; scaling by √d_k keeps them in a range
where softmax gradients are stable (not saturated).

---

## 10. Vanishing Gradients in RNNs

**Q:** The vanishing gradient problem in RNNs occurs primarily because:

- a) Hidden states are normalized.
- b) Softmax saturates.
- **c) Jacobian matrices are repeatedly multiplied during backpropagation ✅**
- d) Parameters change at every time step.

**Explanation:** Backprop through time multiplies the same Jacobian across many steps;
eigenvalues < 1 shrink the gradient toward zero (vanishing).

---

## 11. Transformers vs RNNs

**Q:** Which statement best distinguishes Transformers from RNNs?

- a) Transformers share parameters across vocabulary instead of time.
- b) Transformers cannot model sequences.
- c) Transformers eliminate hidden representations.
- **d) Transformers replace recurrence with attention mechanisms ✅**

**Explanation:** Transformers drop sequential recurrence entirely and model dependencies with
(self-)attention, enabling parallel computation over the sequence.

---

## 12. Linear Predictive Coding (LPC)

**Q:** In Linear Predictive Coding (LPC), each future sample is approximated as:

- **a) A weighted linear combination of previous samples ✅**
- b) A nonlinear function of all future samples.
- c) A random Gaussian variable.
- d) A weighted average of all observations.

**Explanation:** LPC predicts x_t ≈ Σ a_i·x_{t-i} — a **linear** autoregressive combination of
past samples.

---

## 13. Parameter Sharing Across Time in RNNs

**Q:** Parameter sharing across time in an RNN primarily allows:

- a) Reduction of vocabulary size.
- **b) Processing sequences of arbitrary length ✅**
- c) Elimination of hidden states.
- d) Parallel computation across all time steps.

**Explanation:** Because the same weights are reused at every step, an RNN can be unrolled to
**any** sequence length. (It does *not* enable parallelism — that's a Transformer trait.)

---

## 14. Why a Single Final Encoder State Is Inadequate

**Q:** In the encoder-decoder framework, why was relying solely on the final encoder hidden
state considered inadequate?

- **a) It compresses the entire input sequence into a single vector ✅**
- b) It increases computational complexity.
- c) It cannot represent the vocabulary.
- d) It prevents autoregressive decoding.

**Explanation:** A single fixed-size vector is an information bottleneck for long inputs;
attention was introduced so the decoder can look back at **all** encoder states.

---

## 15. Similarity in Scaled Dot-Product Attention

**Q:** In scaled dot-product attention, the similarity between two tokens is computed using:

- **a) Inner product between Query and Key vectors ✅**
- b) Cosine similarity
- c) Euclidean distance
- d) Manhattan distance

**Explanation:** Token similarity = qᵀk (dot/inner product), then scaled by √d_k and softmaxed.

---

## Answer Key

| Q | Ans | Q | Ans | Q | Ans | Q | Ans |
|---|-----|---|-----|---|-----|---|-----|
| 1 | b | 5 | b | 9  | d | 13 | b |
| 2 | b | 6 | b | 10 | c | 14 | a |
| 3 | a | 7 | b | 11 | d | 15 | a |
| 4 | c | 8 | b | 12 | a |    |   |
