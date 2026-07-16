# 📘 Chapter 4 — Training Models: Exercises & Solutions

---

## 🧩 Exercise 1

**Which linear regression training algorithm can you use if you have a training set with millions of features?**

### ✅ Answer

**Gradient Descent** (specifically Stochastic GD or Mini-batch GD), because they scale linearly with the number of features **O(n)**, while the Normal Equation and SVD are **O(n²)** or **O(n³)**.

---

## 🧩 Exercise 2

**Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?**

### ✅ Answer

Algorithms that use distance or gradient-based optimization suffer:

- Gradient Descent (all types)
- SVM
- KNN
- Logistic Regression
- Neural Networks

**Problem:** The model becomes biased toward features with larger scales, and Gradient Descent takes longer to converge.

**Solution:** Apply Feature Scaling using `StandardScaler` or `MinMaxScaler`.

---

## 🧩 Exercise 3

**Can gradient descent get stuck in a local minimum when training a logistic regression model?**

### ✅ Answer

**No.** Because the cost function (log loss) is **convex**, meaning it has only one global minimum and no local minima.

---

## 🧩 Exercise 4

**Do all gradient descent algorithms lead to the same model, provided you let them run long enough?**

### ✅ Answer

**No.**

- **Batch GD** → converges to the exact same model.
- **Stochastic GD** and **Mini-batch GD** → get close but keep bouncing around the minimum (do not converge exactly).

To make them converge, use a **learning schedule**.

---

## 🧩 Exercise 5

**Suppose you use batch gradient descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?**

### ✅ Answer

The **learning rate is too high**, causing the algorithm to diverge.

**Fix:** Reduce the learning rate or use a learning schedule.

---

## 🧩 Exercise 6

**Is it a good idea to stop mini-batch gradient descent immediately when the validation error goes up?**

### ✅ Answer

**No.** Because Mini-batch GD is stochastic, validation error naturally fluctuates. Stopping immediately may stop training too early.

**Better approach:** Use **early stopping with patience** and keep the best model.

---

## 🧩 Exercise 7

**Which gradient descent algorithm will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?**

### ✅ Answer

| Algorithm | Behavior |
|---|---|
| **Stochastic GD** | Fastest to reach the vicinity |
| **Batch GD** | Actually converges to the minimum |
| **Stochastic & Mini-batch GD** | Use a **learning schedule** to make them converge |

---

## 🧩 Exercise 8

**Suppose you are using polynomial regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?**

### ✅ Answer

This is **Overfitting** (high variance).

**Solutions:**

1. Simplify the model (reduce polynomial degree)
2. Add regularization (Ridge, Lasso, Elastic Net)
3. Get more training data

---

## 🧩 Exercise 9

**Suppose you are using ridge regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?**

### ✅ Answer

This is **high bias (underfitting)**.

**Solution:** **Reduce α** (less regularization) to allow the model to fit the data better.

---

## 🧩 Exercise 10

**Why would you want to use:**

### ✅ Answer

| Technique | Reason |
|---|---|
| **Ridge** instead of plain Linear Regression | To reduce overfitting and improve generalization |
| **Lasso** instead of Ridge | To perform automatic feature selection (eliminates unimportant features) |
| **Elastic Net** instead of Lasso | When there are correlated features; it handles them better than Lasso |

---

## 🧩 Exercise 11

**Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two logistic regression classifiers or one softmax regression classifier?**

### ✅ Answer

Use **two logistic regression classifiers** because this is a **multilabel classification** problem (each image can have both labels simultaneously). Softmax is for **mutually exclusive** classes only.

---

## 🧩 Exercise 12

**Implement batch gradient descent with early stopping for softmax regression without using Scikit-Learn (only NumPy). Use it on a classification task such as the iris dataset.**

### ✅ Answer — Implementation

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode labels
y_onehot = np.eye(3)[y]

# Split and scale
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Add bias term
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_val_b   = np.c_[np.ones((X_val.shape[0],   1)), X_val]

# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Batch Gradient Descent with Early Stopping
def softmax_gd(X, y, X_val, y_val, learning_rate=0.1, n_epochs=1000, patience=10):
    n_features = X.shape[1]
    n_classes  = y.shape[1]
    theta = np.random.randn(n_features, n_classes) * 0.01

    best_val_loss = float('inf')
    best_theta    = theta.copy()
    wait          = 0

    for epoch in range(n_epochs):
        # Forward pass
        logits = X @ theta
        probs  = softmax(logits)

        # Loss (cross-entropy)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))

        # Gradient
        gradients = (1 / X.shape[0]) * X.T @ (probs - y)

        # Update weights
        theta -= learning_rate * gradients

        # Validation loss
        val_logits = X_val @ theta
        val_probs  = softmax(val_logits)
        val_loss   = -np.mean(np.sum(y_val * np.log(val_probs + 1e-8), axis=1))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_theta    = theta.copy()
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_theta

# Train the model
theta_best = softmax_gd(X_train_b, y_train, X_val_b, y_val)

# Prediction function
def predict(X, theta):
    return np.argmax(softmax(X @ theta), axis=1)

# Evaluate on validation set
y_pred   = predict(X_val_b, theta_best)
y_true   = np.argmax(y_val, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"Validation Accuracy: {accuracy:.2f}")
```

---

*Chapter 4 — Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*