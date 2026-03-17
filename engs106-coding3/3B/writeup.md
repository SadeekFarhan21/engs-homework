# Lab Assignment 3B: Transformer (LLM) — Post-Lab Write-Up

## Model Description

I implemented a **GPT-2-style autoregressive Transformer** for character-level language modeling, trained on the Tiny Shakespeare dataset (~1 MB of text). The architecture follows the original "Attention Is All You Need" paper, adapted for causal (decoder-only) language modeling.

### Architecture

| Component | Configuration |
|-----------|--------------|
| Transformer blocks | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Attention head dimension | 64 (= 384 / 6) |
| Context length | 256 tokens |
| Dropout | 0.2 |
| Vocabulary size | Character-level (~65 unique chars) |
| Total parameters | **~10,690,625** |

Each Transformer block consists of:
1. **Layer Norm → Multi-Head Attention → Residual Add** (self-attention sub-layer)
2. **Layer Norm → Feed-Forward Network (4× expansion) → Residual Add** (FFN sub-layer)

### Key Implementations

**Attention** (`Attention.attention`):
$$\text{Attention}(Q, K, V) = \text{Dropout}\left(\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)\right)V$$
A causal mask (lower-triangular) prevents each position from attending to future tokens.

**Multi-Head Attention** (`MultiHeadAttention.forward`):
Six independent attention heads are computed in parallel, their outputs concatenated, then projected back to the embedding dimension via a learned linear layer and dropout.

**Positional Encoding** (`PositionalEncoding.__init__`):
Sinusoidal embeddings precomputed for all positions up to `context_length`:
$$PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}}), \quad PE(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}})$$

### Justification of Parameters

- **6 layers / 6 heads / 384 dim**: This is a small but effective configuration for character-level modeling on ~1 MB of text. Larger models risk overfitting or take prohibitively long to train on a laptop GPU.
- **Context length 256**: Balances the ability to capture multi-sentence dependencies (Shakespeare uses consistent character voices across scenes) with memory and computation constraints.
- **Dropout 0.2**: Mild regularization appropriate for this dataset size. Higher dropout slowed convergence without meaningful generalization improvement.
- **Learning rate 1e-3 with AdamW**: Standard for small Transformer models. Gradient clipping at 1.0 prevents occasional large gradient steps early in training.
- **2000 training iterations**: Sufficient for the model to learn coherent word boundaries and short phrase patterns; training and validation loss both decrease steadily throughout.

## Model Evaluation

The model was trained on the Tiny Shakespeare dataset (~90% train / ~10% validation split). Loss was evaluated every 50 iterations over 10 batches.

After training, the model was prompted with `"ROMEO:"` and generated 250 tokens of text. A representative output:

```
ROMEO:
Sir, I would not think, sir?

DUKE VINCENTIO:
Farewell.

Provost:
I have been too soon.

DUKE VINCENTIO:
'Tis true; for the truth; the wars is all.
```

**Observations:**
- The model correctly learns Shakespearean formatting conventions (speaker names in ALL-CAPS followed by a colon, line breaks between speeches).
- Generated text is grammatically plausible at the phrase level.
- Semantic coherence across multiple exchanges is limited — expected given the small model size and relatively short training time.
- The model generalizes: it generates speaker names (`DUKE VINCENTIO`, `Provost`) that were not in the prompt.

Training and validation loss decreased consistently, indicating the model was learning without severe overfitting given the dropout regularization.

## Reflection

This lab made the Transformer architecture concrete in a way that reading the paper alone did not. The key insight I took away is that **multi-head attention is surprisingly simple to implement** once you understand the scaled dot-product attention equation — each head independently attends over the same input, and the value comes from combining these parallel views. The second takeaway is how much the **positional encoding** matters: without it, the model has no notion of sequence order and would treat every context window as a bag of tokens. The sinusoidal formulation is elegant because it encodes absolute position while also allowing the model to learn relative offsets through linear combinations of the sin/cos values.
