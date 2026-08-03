# Chapter 5 – Support Vector Machines (SVM)

## 📘 Overview

Support Vector Machines (SVM) are powerful and versatile models capable of performing linear and nonlinear classification, regression, and even outlier detection. They work well with small to medium-sized nonlinear datasets.

---

## 🧠 Key Concepts

### Linear SVM Classification
- **Idea:** Find the line (or hyperplane) that separates classes with the **widest possible margin**.
- **Support Vectors:** Data points closest to the decision boundary; they determine the margin.
- **Large Margin Classification:** Maximizing the margin improves generalization.

### Hard vs Soft Margin
- **Hard Margin:** No points allowed inside the margin.
  - Works only if data is linearly separable.
  - Sensitive to outliers.
- **Soft Margin:** Allows some margin violations (controlled by hyperparameter **C**).
  - More flexible and works with real-world noisy data.
  - Smaller C → wider margin → simpler model (less overfitting).

---

## 📈 Nonlinear SVM

### Polynomial Features
- Add polynomial features (e.g., x²) to make data linearly separable.
- Problem: Can cause **combinatorial explosion** of features.

### Polynomial Kernel
- Uses **kernel trick** to get the same result as adding polynomial features without actually adding them.
- Hyperparameters: `degree`, `coef0`, `C`.

### Similarity Features (Gaussian RBF)
- Add features that measure similarity to landmarks using Gaussian RBF:
  - `exp(-γ × distance²)`
- Landmarks are chosen from data points.
- Computationally expensive if done directly.

### Gaussian RBF Kernel
- Applies the kernel trick to similarity features.
- **γ (gamma):** Controls the influence of each training instance.
  - Large γ → narrow influence → overfitting.
  - Small γ → wide influence → underfitting.
- **C:** Controls regularization (similar to other models).

---

## 🆚 SVM Classes in Scikit-Learn

| Class | Time Complexity | Out-of-core | Kernel Trick |
|-------|-----------------|-------------|--------------|
| `LinearSVC` | O(m × n) | ❌ No | ❌ No |
| `SVC` | O(m² × n) to O(m³ × n) | ❌ No | ✅ Yes |
| `SGDClassifier` | O(m × n) | ✅ Yes | ❌ No |

---

## 📊 SVM Regression

- **Goal:** Fit as many instances **inside** the margin (street) as possible.
- **Epsilon (ε):** Controls margin width.
  - Smaller ε → more sensitive (more support vectors).
  - Larger ε → wider margin (less sensitive).
- **LinearSVR:** Linear regression with SVM.
- **SVR:** Nonlinear regression using kernel trick.

---

## 🧠 Under the Hood

### Decision Function
`decision = w·x + b`

- If `decision >= 0` → class 1, else class 0.

### Margin Size
`Margin = 2 / ||w||`

- Smaller `w` → wider margin.

### Hinge Loss
- Used to train SVMs with gradient descent.
- **Hinge Loss:** `max(0, 1 - t·s)`
- **Squared Hinge Loss:** `max(0, 1 - t·s)²`
  - LinearSVC uses squared hinge loss by default.
  - SGDClassifier uses hinge loss.

### Dual Problem
- The SVM optimization problem can be solved in **primal** or **dual** form.
- The **dual form** is faster when features > instances.
- The dual form enables the **kernel trick**.

### Kernel Trick
- Allows SVM to work in high-dimensional space **without explicitly transforming data**.
- Common kernels:
  - **Linear:** `K(a,b) = a·b`
  - **Polynomial:** `K(a,b) = (γa·b + r)^d`
  - **RBF:** `K(a,b) = exp(-γ||a-b||²)`
  - **Sigmoid:** `K(a,b) = tanh(γa·b + r)`

---

## ✅ Key Takeaways

1. **SVM = Large Margin Classification**
2. **Support Vectors** determine the decision boundary.
3. **Soft Margin (C)** controls overfitting.
4. **Kernel Trick** enables nonlinear classification without feature explosion.
5. **RBF Kernel** is the most common choice for nonlinear problems.
6. **SVM Regression** uses epsilon (ε) to define margin width.
7. **SVC** is for classification, **SVR** for regression.
8. **LinearSVC** is fast, but no kernel trick.
9. **SGDClassifier** is fast and supports out-of-core learning.