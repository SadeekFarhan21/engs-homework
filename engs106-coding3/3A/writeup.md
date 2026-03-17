# Lab Assignment 3A: Sparse Autoencoder — Brief Report

## What Was Implemented

I implemented three functions in `lab03A.ipynb`:

1. **`sampleIMAGES()`** — randomly samples 10,000 patches of size 8×8 from 10 natural images (512×512 each), flattens each to a 64-dimensional vector, and normalizes to the range [0.1, 0.9].

2. **`sparseAutoencoderCost()`** — computes the total cost $J_\text{sparse}$ and all gradients via forward pass and backpropagation.

3. **`computeNumericalGradient()`** — estimates gradients numerically using the central difference approximation with $\epsilon = 10^{-4}$.

---

## Obstacles and How I Solved Them

### Obstacle 1: `sampleIMAGES` — avoiding full-image copies

My first attempt used `IMAGES[:,:,i][row:row+8, col:col+8]`, which correctly indexes but creates intermediate arrays. I verified that directly indexing `IMAGES[row:row+patchsize, col:col+patchsize, img_idx]` avoids copying the full image every iteration, keeping the function fast.

### Obstacle 2: Backpropagation — including the sparsity gradient in `delta2`

The trickiest part of `sparseAutoencoderCost` was computing the gradient of the KL-divergence sparsity penalty with respect to the hidden activations. The penalty term contributes an additive correction to the hidden-layer delta:

$$\frac{\partial J_\text{sparse}}{\partial a_j^{(2)}} = \beta\left(-\frac{\rho}{\hat\rho_j} + \frac{1-\rho}{1-\hat\rho_j}\right)$$

I initially forgot to add this term to `delta2` before multiplying by the sigmoid derivative, which made the gradient incorrect even though the cost value was right. Adding `sparsity_delta` to the backpropagated signal from layer 3 fixed this:

```python
delta2 = (np.dot(delta3, W2.T) + sparsity_delta) * a2 * (1 - a2)
```

I confirmed correctness by running gradient checking (numerical vs. analytical difference < $10^{-9}$).

### Obstacle 3: Gradient checking was slow

Running `computeNumericalGradient` on the full parameter vector (2×64×25 + 2×25 + 2×64 = 3450 parameters) with the full 10,000-patch dataset takes a very long time (one forward pass per parameter, twice). I debugged using a reduced setup (10 samples, 2 hidden units) and only ran the full check once to confirm correctness before switching to L-BFGS.

### Obstacle 4: Numerical stability in KL divergence

The KL term $\rho \log(\rho / \hat\rho)$ blows up if $\hat\rho \to 0$. I added a small $\epsilon = 10^{-10}$ to the denominator to avoid `nan` values early in training when some hidden units are completely inactive:

```python
kl_div = np.sum(rho * np.log(rho / (rho_hat + 1e-10)) + ...)
```

---

## Results

After training for 2000 L-BFGS iterations:

| Metric | Value |
|--------|-------|
| Untrained cost | 48.86 |
| Trained cost | 0.45 |
| L1 reconstruction error (100 patches) | ~0.4 |

The trained weight filters (`W1.T`) visualize as localized, oriented edge detectors — resembling Gabor filters and the simple cells found in the primary visual cortex (V1). This matches the expected result: a sparse autoencoder trained on natural image patches learns edge-like features because edges are the sparse, independent components of natural images.
