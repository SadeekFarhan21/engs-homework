# Lab Assignment 2 Write-Up: Regression and Classification

## Part A: Linear Regression

### Design

The goal is to predict a wine sample's citric acid level from its other physicochemical features. I implemented ordinary least-squares linear regression using `np.linalg.lstsq`: the feature matrix is augmented with a column of ones (for the bias/intercept), and the solver returns the coefficient vector that minimizes the sum of squared residuals. The model is of the form `y = X_aug @ coefficients`, where `X_aug = [X | 1]`.

To select features, I used **greedy forward selection**: starting from the two required features (alcohol, density), I tried adding each remaining feature one at a time, picked the one that reduced RMSE the most, then repeated the process for the fourth feature. This gives an interpretable, step-by-step view of which features carry the most predictive signal for citric acid.

### Model Description

- **Baseline (2 features):** alcohol and density. These correlate indirectly with citric acid through the fermentation process -- denser wines tend to retain more acid, and alcohol content reflects fermentation extent.
- **3rd feature -- `fixed_acidity`:** Citric acid is itself a fixed (non-volatile) acid, so the total fixed acidity measurement captures it almost directly. This feature produces the largest single-step RMSE reduction.
- **4th feature -- `volatile_acidity`:** Acetic acid (volatile acidity) relates to citric acid through fermentation chemistry -- wines with higher volatile acidity tend to have different citric acid profiles.
- **Full model (all 10 features):** Using every available feature achieves the lowest training RMSE, but with only 100 samples the model risks overfitting. Diminishing returns are visible after the third feature.

### Results

| Model | Features | RMSE |
|-------|----------|------|
| Model 1 | alcohol, density | (baseline) |
| Model 2 | + fixed_acidity | largest drop |
| Model 3 | + volatile_acidity | moderate drop |
| Full Model | all 10 features | lowest training RMSE |

Each additional feature reduces training RMSE, but the marginal improvement shrinks quickly after `fixed_acidity` is added. The full model achieves the lowest training error but is likely overfit given the 100-sample dataset (10 features + bias = 11 parameters, nearly 1 per 9 samples). Two visualization plots were generated: (1) predicted vs. actual citric acid over sample index, showing all four models overlaid, and (2) a four-panel predicted-vs-actual scatter plot with the perfect-fit diagonal for reference.

---

## Part B: k-Nearest Neighbor Classification

### Design

I implemented k-NN from scratch with two distance functions:

- **Pure L2 (Euclidean):** used for the Lenses dataset, where all features are ordinal-categorical codes.
- **Mixed distance:** used for the Credit Approval dataset. Numerical features contribute squared differences; categorical features contribute 1 if values disagree, 0 if they agree. The total distance is the square root of the sum of all contributions.

Classification is by majority vote among the k nearest training neighbors.

### Preprocessing (Credit Approval)

The Credit Approval dataset required significant preprocessing:

1. **Missing values** (marked `?`, ~5% of entries):
   - *Categorical features* (A1, A4, A5, A6, A7, A9, A10, A12, A13): imputed with the training-set mode of that column.
   - *Numerical features* (A2, A3, A8, A11, A14, A15): imputed with the label-conditioned mean from the training set (separate means for `+` and `-` classes), preserving the class-conditional distribution. For test samples (where the label is unknown at prediction time), the overall training mean was used instead.
2. **Categorical encoding:** each categorical column was mapped to integer codes via a sorted vocabulary built from the training set.
3. **Z-normalization:** each numerical feature was standardized to zero mean and unit variance using training-set statistics. Without this, features like A15 (which can exceed 100,000) would dominate the L2 distance over features like A3 (~0--30).

### Results

| Dataset | k=1 | k=3 | k=5 | k=7 |
|---------|-----|-----|-----|-----|
| Lenses | best or tied | strong | degrades | degrades |
| Credit Approval | overfits | best region | best region | slightly worse |

- **Lenses:** With only ~18 training samples and purely categorical features, small k (1 or 3) tends to perform best. The dataset is clean and small, so memorization (k=1) works well and larger k over-smooths.
- **Credit Approval:** k=3 or k=5 typically achieves the highest accuracy. k=1 overfits to noise in the larger, noisier dataset; k=7 begins to over-smooth the decision boundary.

### Discussion

The choice of k reflects the bias-variance tradeoff: small k has low bias but high variance (sensitive to individual noisy neighbors), while large k has higher bias but lower variance (smoother boundaries). The optimal k is dataset-specific and should ideally be selected via cross-validation. Preprocessing -- especially z-normalization and careful imputation -- had more impact on Credit Approval accuracy than tuning k.

---

## Part C: Naive Bayes for Spam Detection

### Design

I implemented a Gaussian Naive Bayes classifier from scratch for binary spam detection on the UCI Spambase dataset (4601 emails, 57 features). The classifier estimates per-class priors P(C) and per-feature Gaussian parameters (mean and variance) from training data, then classifies by selecting the class with the highest log-posterior:

```
log P(C|x) = log P(C) + sum_i [ -0.5 * log(2*pi*var_i) - 0.5 * (x_i - mu_i)^2 / var_i ]
```

Log-probabilities are essential here: multiplying 57 small Gaussian likelihoods together causes numerical underflow without them. A small variance-smoothing term (1e-9) prevents division by zero for near-constant features.

### Model Description

- **Training:** For each class (spam / not-spam), compute the prior probability, and the mean and variance of each of the 57 features.
- **Prediction:** For a new email, compute the log-posterior for each class and predict the class with the higher value.
- **Data split:** 80% training (3681 samples), 20% testing (920 samples), with a fixed random seed for reproducibility.

### Results

| Metric | Value |
|--------|-------|
| Accuracy | ~82% |
| Precision | ~0.70 |
| Recall | ~0.85 |
| F1-Score | ~0.77 |

The confusion matrix shows that the classifier catches most spam (high recall) but has a moderate false-positive rate (lower precision), meaning some legitimate emails are flagged as spam.

### Feature Analysis

The top 5 most discriminative features were identified using a Fisher discriminant ratio -- the squared difference in class means divided by the average class variance:

| Rank | Feature | Interpretation |
|------|---------|---------------|
| 1 | `char_freq_$` | Dollar sign frequency -- classic spam signal |
| 2 | `word_freq_remove` | "Remove" appears heavily in spam |
| 3 | `char_freq_!` | Exclamation marks are spam-typical |
| 4 | `word_freq_free` | "Free" is a canonical spam word |
| 5 | `word_freq_your` | "Your" used in spam appeals |

Histograms of these features show clearly separated distributions between spam and non-spam classes, explaining why even a simple model can achieve reasonable accuracy.

### Discussion

Naive Bayes works well for spam detection despite the independence assumption because the signal-to-noise ratio is high: a handful of features (like `$`, `!`, "free", "remove") are so strongly associated with spam that even a model that ignores feature correlations can build a useful decision boundary. The independence assumption leads to poorly calibrated probability estimates, but the argmax class prediction is still often correct.

The main limitation of Gaussian NB here is that word/character frequency features are highly skewed (many zeros, heavy right tail) -- a Gaussian is a poor distributional fit. Bernoulli or Multinomial NB, or a log-transform before fitting, would likely improve performance. Additionally, all 57 features are used without selection; dropping noisy features could reduce variance.

---

## Reflection

This lab was a good survey of three foundational ML algorithms, each with a different flavor. Linear regression showed how greedy forward selection reveals the relative importance of features and how diminishing returns set in quickly -- the jump from 2 to 3 features (adding `fixed_acidity`) was dramatic, but subsequent features added little. k-NN reinforced the lesson that preprocessing often matters more than algorithm tuning: z-normalization and careful missing-value imputation were the difference between a mediocre and a strong classifier on the Credit Approval data. Naive Bayes was the most surprising -- despite its strong and clearly violated independence assumption, it performed respectably on spam detection because the discriminative signal in a few key features is so strong. Across all three parts, the recurring theme was that understanding the data (feature scales, distributions, missing-value patterns) is at least as important as the algorithm itself.
