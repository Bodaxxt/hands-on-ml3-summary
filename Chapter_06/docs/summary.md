# 🌲 Chapter 6: Decision Trees (أشجار القرار)

> **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**  
> Comprehensive study notes, visual explanations, and core concepts for Chapter 6.

---

## 📑 Table of Contents (فهرس المحتويات)
- [1. Overview & Intuition](#1-overview--intuition)
- [2. How Decision Trees Work](#2-how-decision-trees-work)
- [3. Training Algorithm: CART](#3-training-algorithm-cart)
- [4. Gini Impurity vs. Entropy](#4-gini-impurity-vs-entropy)
- [5. Decision Trees for Regression](#5-decision-trees-for-regression)
- [6. Computational Complexity](#6-computational-complexity)
- [7. Regularization & Preventing Overfitting](#7-regularization--preventing-overfitting)
- [8. Limitations & Instabilities](#8-limitations--instabilities)
- [9. Summary Cheat Sheet](#9-summary-cheat-sheet)

---

## 1. Overview & Intuition
**Decision Trees** are versatile, non-parametric supervised machine learning models capable of performing:
- **Binary & Multi-class Classification**
- **Multi-output Classification**
- **Non-linear Regression**

They form the fundamental building blocks of powerful ensemble models such as **Random Forests**, **Extra-Trees**, and **Gradient Boosted Trees (XGBoost / LightGBM)**.

### ✨ Key Strengths:
- **White Box Model**: Decisions are completely interpretable and explainable.
- **Minimal Data Preparation**: Do **not** require feature scaling (normalization/standardization) or centering.
- **Handles Categorical & Numerical Data**.

---

## 2. How Decision Trees Work

```
                 [ Root Node: Feature X <= Threshold ]
                               /      \
                             Yes       No
                             /          \
               [ Internal Node ]      [ Leaf Node: Class A ]
                    /       \
                  Yes        No
                  /            \
        [ Leaf Node: Class B ]  [ Leaf Node: Class C ]
```

- **Root Node**: The initial split on the whole dataset.
- **Internal Nodes**: Decision rules splitting samples based on feature thresholds (e.g., `petal width <= 0.8 cm`).
- **Leaf Nodes**: Terminal nodes that output the final prediction (class label or continuous value).

### Prediction Mechanism:
1. Start at the root node.
2. Evaluate condition: If true, move left; otherwise, move right.
3. Continue traversing down the tree until reaching a leaf node.
4. Output class probabilities $\hat{p}_{i,k}$ or predicted value $\hat{y}$.

---

## 3. Training Algorithm: CART

Scikit-Learn uses the **CART (Classification and Regression Tree)** algorithm to train Decision Trees.

### How CART Works:
1. **Greedy Approach**: At each step, it searches for the pair $(k, t_k)$ (feature $k$ and threshold $t_k$) that produces the purest subsets (lowest cost).
2. **Cost Function for Classification**:
   $$J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$
   where $G$ is the impurity (Gini or Entropy) and $m$ is the sample count.
3. **Recursive Splitting**: Repeats the binary splitting until reaching stopping criteria (`max_depth`, `min_samples_leaf`, etc.).

> **Note**: CART is a *greedy algorithm* (finds locally optimal splits at each step) and does not guarantee the globally optimal tree (finding an optimal tree is NP-Complete, $\mathcal{O}(\exp(m))$).

---

## 4. Gini Impurity vs. Entropy

| Metric | Formula | Characteristics |
| :--- | :--- | :--- |
| **Gini Impurity** | $G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2$ | Default in `sklearn`. Slightly faster to compute. Leads to frequent class isolation. |
| **Entropy** | $H_i = -\sum_{k=1, p_{i,k}\neq 0}^{n} p_{i,k} \log_2(p_{i,k})$ | Information gain based. Tends to produce slightly more balanced trees. |

In practice, both metrics yield very similar trees, and Gini is preferred as a faster default.

---

## 5. Decision Trees for Regression

Instead of predicting class probabilities, regression trees predict the **average target value** of training instances in each region/leaf.

### CART Regression Cost Function:
$$J(k, t_k) = \frac{m_{\text{left}}}{m} \text{MSE}_{\text{left}} + \frac{m_{\text{right}}}{m} \text{MSE}_{\text{right}}$$
where:
$$\text{MSE}_{\text{node}} = \frac{1}{m_{\text{node}}} \sum_{i \in \text{node}} (\hat{y}_{\text{node}} - y^{(i)})^2, \quad \hat{y}_{\text{node}} = \frac{1}{m_{\text{node}}} \sum_{i \in \text{node}} y^{(i)}$$

---

## 6. Computational Complexity

| Task | Time Complexity | Notes |
| :--- | :--- | :--- |
| **Prediction** | $\mathcal{O}(\log_2(m))$ | Very fast, independent of number of features $n$. |
| **Training** | $\mathcal{O}(n \times m \log_2(m))$ | Scales reasonably well with dataset size $m$. |

---

## 7. Regularization & Preventing Overfitting

Decision Trees are **non-parametric models** that adapt freely to training data without fixed assumptions. Without constraints, they will readily overfit.

### Key Hyperparameters in Scikit-Learn:
- `max_depth`: Limits the tree depth (default: `None`).
- `min_samples_split`: Minimum number of samples required to split a node (e.g., 2, 5, 10).
- `min_samples_leaf`: Minimum number of samples required in a leaf node (e.g., 1, 5).
- `max_leaf_nodes`: Maximum number of terminal leaves allowed.
- `max_features`: Maximum features evaluated for the best split at each node.

> 💡 **Tip**: Increasing `min_*` hyperparameters or decreasing `max_*` hyperparameters regularizes the model and reduces overfitting.

---

## 8. Limitations & Instabilities

1. **Orthogonal Decision Boundaries**: Splits are always perpendicular to an axis, making trees sensitive to orientation/rotations in the feature space (Mitigation: use **PCA** preprocessing).
2. **High Variance & Instability**: Small changes in training data can result in completely different tree structures.
3. **Greedy Traversal**: Cannot backtrack; may miss global optima.

> 🌟 **Solution to Limitations**: Ensemble methods like **Random Forests** average multiple trees to overcome variance and instability!

---

## 9. Summary Cheat Sheet

| Aspect | Decision Trees |
| :--- | :--- |
| **Model Type** | White-Box, Supervised (Classification & Regression) |
| **Feature Scaling** | Not required |
| **Primary Algorithm** | CART (Binary recursive splitting) |
| **Splitting Criteria** | Gini Impurity, Entropy (Classification) / MSE, MAE (Regression) |
| **Main Challenge** | Overfitting (High Variance) |
| **Key Regularizers** | `max_depth`, `min_samples_leaf`, `min_samples_split` |
