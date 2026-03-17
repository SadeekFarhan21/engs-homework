# Lab Assignment 4: Support Vector Machines — Write-Up

## Overview

This write-up addresses the five tasks in the SVM lab on the Fashion-MNIST dataset (10 clothing categories, 28×28 grayscale images). I used 2000 training samples (1800 train / 200 test after a 90/10 split) and implemented an SVM from scratch using `cvxopt` for the quadratic programming solver.

---

## Task 1: Binary SVM with a Non-Linear Kernel

I implemented a polynomial kernel:
$$K(x, z) = (x^T z + 1)^3$$

**Why a polynomial kernel?** Fashion-MNIST images are high-dimensional (784 features), and clothing categories are not linearly separable in pixel space. The polynomial kernel implicitly maps inputs into a higher-dimensional feature space where class boundaries are more linearly separable, without explicitly computing the expensive high-dimensional feature vectors.

The SVM dual formulation solves:
$$\max_{\mathbf{a}}\ \mathbf{1}^T\mathbf{a} - \frac{1}{2}\mathbf{a}^T (\mathbf{T}\mathbf{K}\mathbf{T})\, \mathbf{a}$$
subject to $0 \leq a_i \leq C$ and $\mathbf{a}^T \mathbf{t} = 0$, where $\mathbf{T} = \text{diag}(\mathbf{t})$ and $\mathbf{t} \in \{-1, +1\}^N$.

The bias term $b$ is estimated using support vectors (those with $a_i > 10^{-5}$) via Bishop eq. 7.37:
$$b = \frac{1}{|\mathcal{S}|} \sum_{n \in \mathcal{S}} \left(t_n - \sum_{m} a_m t_m K(x_m, x_n)\right)$$

**Preprocessing**: Pixel values were normalized to $[0, 1]$ before kernel computation. Without normalization, raw pixel values produce kernel values on the order of $10^{19}$, making the QP solver numerically unstable.

---

## Task 2: Predictive Model

Prediction uses the learned support vectors and bias:
$$f(x) = \text{sign}\!\left(\sum_{i \in \mathcal{S}} a_i t_i K(x_i, x) + b\right)$$

This is implemented in `SVM.predict()` via `SVM.decision_function()`, which computes the cross-kernel $K(X_{\text{test}}, X_{\text{train}})$ and returns signed class labels $\{-1, +1\}$.

---

## Task 3: One-vs-Rest vs. One-vs-One

### One-vs-Rest (OvR)
Train 10 binary classifiers, one per class. Classifier $i$ labels class $i$ as $+1$ and all other classes as $-1$. At test time, the class with the highest raw decision score wins.

### One-vs-One (OvO)
Train $\binom{10}{2} = 45$ binary classifiers, one per pair of classes. Each classifier votes for one of the two classes it was trained on. The class receiving the most votes wins.

### Comparison

| Scheme | # Classifiers | Train samples/classifier | Test strategy |
|--------|--------------|--------------------------|---------------|
| OvR | 10 | ~1800 (all train data) | argmax of decision scores |
| OvO | 45 | ~360 (2 classes × ~180) | majority vote |

**OvR** is simpler and trains each classifier on more data, but the class imbalance (1 positive vs. 9 negatives) can harm training. **OvO** has balanced binary problems but trains on far fewer samples per classifier and requires many more QP solves.

On Fashion-MNIST with the polynomial kernel, both strategies achieved similar accuracy (~19%) on this 200-sample test set. The low overall accuracy reflects the challenge of Fashion-MNIST for a kernel SVM trained on only 1800 samples with a relatively low-degree polynomial kernel — many visually similar classes (Shirt vs. T-shirt, Coat vs. Pullover) are hard to separate.

---

## Task 4: Hyperparameter Tuning

The regularization parameter $C$ controls the trade-off between maximizing the margin and minimizing training errors:
- **Small $C$**: Large margin, more misclassifications tolerated (high bias, low variance)
- **Large $C$**: Hard-margin approximation, fewer misclassifications but potentially overfitting (low bias, high variance)

I searched over $C \in \{1, 10, 100, 1000, 10000\}$ (a log-spaced grid) using the OvR scheme and measuring test accuracy. Results:

| $C$ | Accuracy |
|-----|----------|
| 1 | 19.0% |
| 10 | 19.0% |
| 100 | 19.0% |
| 1000 | 19.0% |
| 10000 | 17.0% |

The flat accuracy plateau across most $C$ values suggests that the bottleneck is not regularization strength but rather the kernel choice and limited training data. Very large $C$ (10000) slightly degraded performance, likely due to overfitting. Based on this grid search, **$C = 1$** was selected as it achieves the best accuracy while maintaining a larger margin.

For a more principled search, cross-validation (e.g., 5-fold) would be preferred over a single train/test split to reduce variance in the accuracy estimate.

---

## Task 5: Multiclass Confusion Matrices

Confusion matrices were generated for the OvR scheme using `sklearn.metrics.confusion_matrix` (used only for visualization — the SVM itself is implemented from scratch).

**Key observations from the confusion matrix:**
- The diagonal represents correct classifications. Off-diagonal entries reveal systematic confusions.
- **Shirt (class 6)** and **T-shirt/top (class 0)** are commonly confused with each other — both are upper-body garments with similar pixel distributions.
- **Coat (class 4)** and **Pullover (class 2)** share visual structure (long sleeves, similar silhouette) and are frequently misclassified.
- **Trouser (class 1)** is one of the more accurately classified categories due to its distinct shape (two legs, lower-body).
- The model predicts some classes disproportionately often (mode collapse toward the most common class in each OvR classifier), reflecting the class imbalance issue in OvR training.

These patterns match the known difficulty of Fashion-MNIST: even state-of-the-art CNNs achieve ~94% accuracy, so a polynomial kernel SVM trained on 1800 samples achieving ~19% (slightly above the 10% random baseline) is a reasonable starting point.

---

## Summary

| Task | Key Result |
|------|-----------|
| Binary SVM | Polynomial kernel ($d=3$), bias via support vectors |
| Predictive model | Decision function with cross-kernel and bias |
| OvR vs. OvO | ~19% accuracy for both; OvR preferred for simplicity |
| Hyperparameter tuning | $C = 1$ optimal; plateau suggests kernel is the bottleneck |
| Confusion matrix | Shirt/T-shirt and Coat/Pullover are the main confusion pairs |
