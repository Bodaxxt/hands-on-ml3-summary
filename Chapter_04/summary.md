# Chapter 4 – Training Models

## 📘 Overview
This chapter explains how machine learning models are trained under the hood.

### Key Topics
- **Linear Regression** – Normal Equation, SVD, Gradient Descent
- **Gradient Descent** – Batch, Stochastic, Mini-batch
- **Polynomial Regression** – Adding powers of features
- **Learning Curves** – Diagnosing underfitting and overfitting
- **Regularization** – Ridge (L2), Lasso (L1), Elastic Net
- **Early Stopping** – Stop training when validation error increases
- **Logistic Regression** – Binary classification with sigmoid
- **Softmax Regression** – Multiclass classification

---

## 📊 Comparison of Linear Regression Training Algorithms

| Algorithm | Large m | Out-of-core | Large n | Hyperparams | In Scikit-Learn |
|-----------|---------|-------------|---------|-------------|-----------------|
| Normal Eq | Fast | No | Slow | 0 | No |
| SVD | Fast | No | Slow | 0 | Yes |
| Batch GD | Slow | No | Fast | 2 | No |
| Stochastic GD | Fast | Yes | Fast | ≥2 | Yes |
| Mini-batch GD | Fast | Yes | Fast | ≥2 | No |

---

## 🧠 Gradient Descent Types

| Type | Data per Step | Speed | Precision |
|------|---------------|-------|-----------|
| Batch GD | All data | Slow | High |
| Stochastic GD | 1 instance | Fast | Low (bounces) |
| Mini-batch GD | Small batch | Fast | Medium |

---

## 📈 Regularization Techniques

| Type | Penalty | Effect |
|------|---------|--------|
| Ridge (L2) | Σθ² | Shrinks coefficients |
| Lasso (L1) | Σ|θ| | Eliminates features |
| Elastic Net | L1 + L2 | Mix of both |

---

## ✅ Key Takeaways

1. **Gradient Descent** is the most scalable training method.
2. **Feature Scaling** is critical for GD performance.
3. **Polynomial Regression** handles non-linear data but risks overfitting.
4. **Learning Curves** help detect underfitting/overfitting.
5. **Regularization** reduces overfitting.
6. **Early Stopping** is a simple and effective regularizer.
7. **Logistic Regression** uses sigmoid + log loss for binary classification.
8. **Softmax Regression** extends logistic regression to multiple classes.