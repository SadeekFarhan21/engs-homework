# Lab Assignment 4: Support Vector Machines — Write-Up

## Overview

This write-up addresses the five tasks in the SVM lab on the Fashion-MNIST dataset (10 clothing categories, 28x28 grayscale images). I used 2000 training samples (1800 train / 200 test after a 90/10 split) and implemented an SVM from scratch using `cvxopt` for the quadratic programming solver.

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
| OvO | 45 | ~360 (2 classes x ~180) | majority vote |

**OvR** is simpler and trains each classifier on more data, but the class imbalance (1 positive vs. 9 negatives) can bias the decision boundaries. **OvO** has balanced binary problems but trains on far fewer samples per classifier and requires many more QP solves.

On Fashion-MNIST with the degree-3 polynomial kernel and $C = 1$, OvR achieved **79.5%** and OvO achieved **78.5%** on the 200-sample test set. The two strategies perform comparably, with OvR edging ahead slightly — likely because the decision-score aggregation in OvR is more informative than simple majority voting in OvO. Both results are solid given the constraints: only 1800 training samples, a fixed polynomial kernel, and visually similar classes (Shirt vs. T-shirt, Coat vs. Pullover).

---

## Task 4: Hyperparameter Tuning

The regularization parameter $C$ controls the trade-off between maximizing the margin and minimizing training errors:
- **Small $C$**: Large margin, more misclassifications tolerated (high bias, low variance)
- **Large $C$**: Hard-margin approximation, fewer misclassifications but potentially overfitting (low bias, high variance)

Following the assignment hint to try high $C$ values, I searched over a log-spaced grid $C \in \{10, 21.5, 46.4, 100\}$ (via `np.logspace(1, 2, 4)`) using the OvR scheme. Results:

| $C$ | Accuracy |
|-----|----------|
| 10.00 | 79.5% |
| 21.54 | 79.5% |
| 46.42 | 79.5% |
| 100.00 | 79.5% |

Accuracy was stable across the entire range, indicating that the model is not sensitive to $C$ in this regime. This makes sense: the polynomial kernel already provides enough capacity to separate the classes, and once $C$ is large enough to allow the solver to fit the training data well, further increases have no effect. Since all values tied, **$C = 10$** was selected as it achieves the same accuracy with a slightly larger margin (more regularization).

For a more principled search, cross-validation (e.g., 5-fold) would be preferred over a single train/test split to reduce variance in the accuracy estimate.

---

## Task 5: Multiclass Confusion Matrices

Confusion matrices were generated for the OvR scheme using `sklearn.metrics.confusion_matrix` (used only for visualization — the SVM itself is implemented from scratch).

**Key observations from the confusion matrix:**
- The diagonal represents correct classifications. Most classes are classified correctly at a high rate, with the majority of predictions falling on the diagonal.
- **Shirt (class 6)** and **T-shirt/top (class 0)** are the most commonly confused pair — both are upper-body garments with similar pixel distributions.
- **Coat (class 4)** and **Pullover (class 2)** share visual structure (long sleeves, similar silhouette) and are also occasionally misclassified as each other.
- **Trouser (class 1)**, **Sandal (class 5)**, **Sneaker (class 7)**, and **Bag (class 8)** are classified more reliably due to their distinct silhouettes — these categories look nothing like upper-body clothing.

These confusion patterns match the known difficulty structure of Fashion-MNIST. Visually similar upper-body garments account for the majority of errors, while categories with distinctive shapes are well-separated by the polynomial kernel.

---

## Summary

| Task | Key Result |
|------|-----------|
| Binary SVM | Polynomial kernel ($d=3$), bias via support vectors |
| Predictive model | Decision function with cross-kernel and bias |
| OvR vs. OvO | 79.5% (OvR) vs. 78.5% (OvO); OvR preferred for simplicity |
| Hyperparameter tuning | $C = 10$ selected; accuracy stable across [10, 100] |
| Confusion matrix | Shirt/T-shirt and Coat/Pullover are the main confusion pairs |
