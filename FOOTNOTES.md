# Project Notes (streamlined, complete)

## 0) What I built (status)

Implemented from scratch (each with KV cache support):

* **GPT-2 byte-level BPE tokenizer**
* **GPT-2 model**
* **Llama-3 style transformer**
* **Gemma-3 style variant**
* **Qwen-style variant**

---

## 1) Baseline mental model: transformer skeleton (the shared core)

A useful way to think about the transformer is:

* **Residual stream** is the main “state” carried forward.
* **Attention** is the **information-sharing / message passing** module: tokens read from other tokens and write back an update into the stream.
* **MLP** is the **computation** module: mostly token-local compute on the residual stream (nonlinear routing / gating).
* Each block is basically:
  **(Norm → Attention → Add)** then **(Norm → MLP → Add)** (exact norm placement differs by model family).

### Attention: what heads really do

* Each head has its own learned projection (“lens”): it looks at the same residual stream but through a different feature subspace.
* A head computes which other tokens matter (attention weights), then **pulls a weighted blend** of their values into the current token.
* Heads are **independent while reading**, then **mixed together afterward** via the output projection and added back to the residual stream.
* More heads = more distinct interaction patterns (good inductive bias).
* Bias terms: GPT-2 uses them; Llama-family variants often remove them (bias effects are often redundant with residual + normalization).

### MLP: why nonlinearity / gating matters

* Stacked linear layers collapse to one linear map → no conditional behavior.
* Nonlinearity + gating enables input-dependent routing: “if pattern present, amplify/pass; otherwise suppress.”

---

## 2) Tokenizer checkpoint: GPT-2 byte-level BPE (from scratch)

Goal: make BPE work **losslessly** over arbitrary text.

### Encoding pipeline (training + inference)

1. **Text → UTF-8 bytes**
2. **Bytes → unique Unicode stand-ins** (1–1 mapping for 256 byte values)
   This makes BPE merges operate over Unicode tokens while still being reversible.
3. **Optional GPT-2 regex pre-split**
   Splits text into chunks so merges are confined within chunks; handles leading spaces + punctuation behavior properly.
4. **Initialize vocab** with the 256 stand-ins (each mapped to a token id)
5. **Train merges**

   * Count most frequent adjacent pairs **within each regex chunk**
   * Merge top pair into a new token
   * Repeat until target vocab size
6. **Encode**

   * Apply merges greedily by merge rank
7. **Decode**

   * Invert stand-ins → bytes → UTF-8

### Library note (important for “real” inference compatibility)

Tokenizer equivalence is not just “BPE vs not BPE.” Real tokenizers include:

* normalization, pre-tokenization, merge rules, special tokens, post-processing, etc.

So:

* For **inference**, usually prefer **`tiktoken`** (fast BPE) or **`tokenizers`**
* **sentencepiece** is often more of a **training pipeline** tool (but can be used for inference if the model expects it)
* Best rule: **match the model creator’s reference tokenizer exactly**

---

## 3) Positional encoding story (the big conceptual pivot)

### Absolute vs relative (what changes conceptually)

* **Absolute position encodings (APE)**: add a position-dependent vector to token embeddings.
  Pros: simple. Cons: brittle to insertions/shift; doesn’t inherently represent distance.
* **Relative position approaches**: encode *distances* between tokens, so attention behavior is shift-invariant.

### Two injection locations (where position enters)

1. **Add to embeddings (APE)**: position is in the input representation.
2. **Modify attention (MAM = modify attention matrix/scores)**: inject position directly into attention interactions (bias/rotation/etc).
   This is especially natural for relative position because attention is inherently pairwise.

### Practical notes I care about

* Relative encodings can better reflect linguistic structure, but may cost extra machinery/params and aren’t a universal win.
* Models differ by:

  * learned vs fixed encodings
  * whether they extrapolate to longer contexts well
* Desired traits (what I’m optimizing for conceptually):

  * continuous / smooth / stable signal
  * multi-scale encoding
  * relative position recoverability
  * generalization to arbitrary length
  * empirically: often better under certain context regimes (esp. when long-context stability matters)

---

## 4) RoPE intuition (the “why this works” explanation I’m using)

We want a positional encoding function that, in attention, depends on:

* the token embeddings
* their **relative** positions
  and not require “hard-coded absolute index storage” inside embeddings (which can get washed out).

Key observation:

* Dot products preserve information about **relative geometric relationships**.
* Rotations are simple linear transforms that preserve norms and angles.

### RoPE in one narrative

* Treat (pairs of) embedding dimensions like **complex numbers / 2D planes**.
* Encode position by rotating queries/keys by an angle that increases **linearly** with position.
* If a token is shifted by Δ positions, its q/k gets rotated by an additional Δ-dependent angle, so **q·k naturally reflects relative distance**.

### Why multi-frequency bands are essential

If every subspace rotated at one speed, position becomes a single periodic pattern → aliasing.
So RoPE assigns **different rotation frequencies** across embedding subspaces:

* some dims rotate fast (local relationships)
* others rotate slow (long-range relationships)

Result:

* **multi-scale positional signal**
* smooth extrapolation
* relative distances show up directly in q/k dot products without adding absolute embeddings

---

## 5) Checkpoint models (what differs, in the way I’m organizing it)

### 5.1 GPT-2 (classic baseline)

Core traits:

* Token embedding + **learned absolute positional embedding**
* LayerNorm (zero-center + unit variance + affine)
* Standard MHA + MLP
* Causal mask as a buffer
* Positions computed at runtime (e.g. `arange`)

Why it’s my baseline:

* Clean reference for KV cache mechanics, attention/logit inspection, and tokenizer integration.
* Conventional head geometry: head_dim aligns in the usual way with d_model / n_heads.
* Uses biases (contrasts with later bias-free families).

---

### 5.2 Llama-3 style (scaling-era baseline)

Core changes relative to GPT-2:

* **RoPE applied to q/k** (no learned position embeddings)
* **RMSNorm** instead of LayerNorm
* Bias-free attention/MLP (common in this family)
* **Attention math up to softmax is done in float32**

  * qk matmul accumulation / softmax stability
  * RoPE sin/cos and application also done in float32
  * (Open question I noted for myself: RMSNorm in fp32 in some implementations, esp. for stability)

Interpretation:

* Still the same KV cache story as GPT-2, but cleaner norms + no bias.
* Position now “lives where it matters” (q/k interactions), not as an additive embedding.

---

### 5.3 Gemma-3 style (very long context + locality tricks)

High-level goal: push to very long contexts while staying stable and not exploding KV cache.

Key traits I noted:

* **Sliding-window attention** for most layers, plus **a few global attention layers**

  * local layers can truncate cache (windowed KV)
  * global layers preserve long-range routing occasionally
* **Extra normalization**:

  * “sandwich” norms (more LNs around attn/MLP paths)
  * **q/k normalization** (pre-attention)
  * I also saw “4 layer norms” described as: pre/post around both attn and MLP, plus consecutive norms (implementation detail but matters for stability)
* **MLP nonlinearity detail**:

  * tanh approximation used in some Gemma-3 implementations (tanh-GELU-ish speed trick)
* **Scaling quirks**

  * embedding scaled by `sqrt(d_model)`-like factor (your note: `self.emb_dim * 0.5` multiplier)
  * also scaling query before attention
* **Attention head style**

  * MQA (or GQA/MQA flavor): fewer KV heads than query heads → reduces KV cache memory footprint
* **Dim/Vocab notes you captured**

  * vocab size roughly **double** (vs GPT-2 baseline framing)
  * embed dim ~**640**, FF ~**2048**
* **RoPE config**

  * theta / RoPE settings may differ between local vs global layers (noted as a design knob)

---

### 5.4 Qwen-style (long context engineering fork)

Main theme: long context (e.g. ~32k) stabilized by extra norms + head geometry changes.

Key traits you listed:

* Long context achieved partly by **curriculum**: train short → gradually increase length.
* **RMSNorm on q and k before RoPE** (extra stability for long context)
* d_model is **smaller** than Llama-3-style (thin) but FF ratio is larger (“tall MLP”)

  * you noted ~2.7× vs Llama’s common ~4×
* Head geometry is more “explicit”:

  * think in terms of **n_heads**, **n_kv_heads**, **head_dim**
  * q projection sometimes maps d_model → **2×d_model**, making head_dim bookkeeping less “standard”
  * open question you recorded: how exactly head_dim is chosen when q width changes; whether n_kv_heads * head_dim ties back to d_model in the usual way (not always)
* Vocab similar to Llama-family (per your note)
* Emphasis on cache efficiency + “thinking mode” behavior (post-training / behavior layer more than core architecture)

---

## 6) Tensor/device/dtype hygiene (what I learned building this)

### Tensor categories (practical mental model)

* **Parameters**: weights/biases, require grads, live in state_dict
* **Buffers**:

  * persistent buffers: saved in state_dict (e.g., running stats or masks if you choose)
  * non-persistent buffers: not in state_dict; useful for lazily created masks, caches, etc
* **Transient / on-the-fly tensors**: created per forward (pos indices, causal masks shaped to T, temporary activations)

Examples:

* GPT-2 position indices:

  * `pos = arange(T_past, T_total, device=toks.device).unsqueeze(0)`
* Causal mask is typically a buffer (or lazily created tensor cached by shape/device/dtype).

### Why lazy-init buffers matters

If buffers are eagerly created on one device/dtype, moving models across devices becomes painful.
Better: lazily init buffers after model is placed and dtype is set, so buffers inherit correct device/dtype.

---

## 7) Loading + quantization workflow (my current recipe)

### My current “safe” loading flow

1. Define model (track params, buffers, transient tensors). Don’t eagerly init device/dtype-sensitive buffers.
2. Init model on **meta** (device=meta).
3. Set dtype (e.g., fp16). Floats change dtype; ints/bools remain safe.
4. Swap/replace layers if doing quantization (layer dtype/behavior may differ).
5. `to_empty(target_device)` to materialize structure on GPU without allocating full weight storage eagerly.
6. Load weights layer-by-layer from checkpoint:

   * allow quant layers to handle custom load if needed
   * otherwise do direct copy

### Quantization landscape (the buckets I’m using)

**Weight-only quantization**

* naive int8 linear that dequantizes outside compute kernel: saves memory, may hurt speed
* int8 linear with dequant inside kernel: saves memory *and* can reduce latency
* activation-aware weight-only schemes: AWQ / GPTQ family (use calibration)

**Activation quantization**

* SmoothQuant, etc.
* tends to be more hardware-dependent (needs specialized int cores to shine)
* good for “long input, short output” workloads (e.g., RAG style)

Tooling note you captured:

* bitsandbytes often materializes on CPU then packs to GPU
* torchao can be more direct about loading/packing

---

## 8) Precision notes (important implementation gotchas)

* Llama-3: attention qk→softmax path is float32; RoPE sin/cos and apply also float32 (even if weights are fp16)
* You suspected RMSNorm might be in fp32 in some families/impls (esp. Gemma-3) for stability

---

## 9) Interpretability hooks (things to check during debugging)

* **Probability trees**: compare chosen branch vs high-prob rejected branches to see uncertainty/mode-switching.
* **Residual trajectories**: PCA basis on embeddings, track token vectors across layers (representation drift / steering).
* **Attention patterns**:

  * sinks (columns saturating)
  * induction bands (off-diagonal repeats)
  * copy diagonals
    Pair with logits + residual stream changes to explain token choices.

---