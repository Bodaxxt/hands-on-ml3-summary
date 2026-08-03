# Solutions – Chapter 5 Exercises

## 🧩 Exercise 1
**What is the fundamental idea behind support vector machines?**

**Answer:**  
SVM finds the decision boundary (hyperplane) that separates classes with the **maximum possible margin** (distance from the boundary to the nearest data points of each class).

---

## 🧩 Exercise 2
**What is a support vector?**

**Answer:**  
A support vector is a data point that lies closest to the decision boundary. These points determine the margin and the position of the decision boundary. Removing or changing them alters the model.

---

## 🧩 Exercise 3
**Why is it important to scale the inputs when using SVMs?**

**Answer:**  
SVM relies on distance calculations. If features have different scales, the model becomes biased toward features with larger values. Scaling ensures all features contribute equally.

---

## 🧩 Exercise 4
**Can an SVM classifier output a confidence score? What about a probability?**

**Answer:**  
- **Confidence score:** Yes, using `decision_function()` (signed distance from the boundary).  
- **Probability:** Not in `LinearSVC`. In `SVC`, set `probability=True` to get probabilities (but training becomes slower).

---

## 🧩 Exercise 5
**How can you choose between LinearSVC, SVC, and SGDClassifier?**

**Answer:**
- **LinearSVC:** Linear data, medium/large datasets, no kernel needed.
- **SVC:** Nonlinear data, small/medium datasets, needs kernel.
- **SGDClassifier:** Very large datasets, out-of-core learning, linear model.

---

## 🧩 Exercise 6
**If an RBF SVM underfits, should you increase or decrease γ and C?**

**Answer:**  
**Increase both.** Underfitting means the model is too simple. Increasing γ makes the RBF kernel more local, and increasing C reduces regularization → more complex model.

---

## 🧩 Exercise 7
**What does it mean for a model to be ϵ-insensitive?**

**Answer:**  
In SVM regression, errors smaller than `ϵ` are ignored. The model does not care about points inside the margin (street).

---

## 🧩 Exercise 8
**What is the point of using the kernel trick?**

**Answer:**  
The kernel trick allows SVM to work in high-dimensional space **without explicitly transforming the data**, avoiding the computational cost and feature explosion.

---

## 🧩 Exercise 9
**Train LinearSVC, SVC, and SGDClassifier on a linearly separable dataset and compare.**

```python
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_blobs(n_samples=100, centers=2, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linear_svc = LinearSVC(C=1, max_iter=10000, random_state=42)
svc = SVC(kernel='linear', C=1, random_state=42)
sgd = SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000, random_state=42)

linear_svc.fit(X_scaled, y)
svc.fit(X_scaled, y)
sgd.fit(X_scaled, y)

print("LinearSVC accuracy:", linear_svc.score(X_scaled, y))
print("SVC accuracy:", svc.score(X_scaled, y))
print("SGD accuracy:", sgd.score(X_scaled, y))
```

---

## 🧩 Exercise 10
**Train SVM on the wine dataset (multiclass).**

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
svm_clf.fit(X_train, y_train)
print("Test accuracy:", svm_clf.score(X_test, y_test))  # ~98-100%
```

---

## 🧩 Exercise 11
**Train and fine-tune an SVR on California housing.**

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_small = X_train[:2000]
y_train_small = y_train[:2000]

svr_pipe = make_pipeline(StandardScaler(), SVR())
param_grid = {
    'svr__kernel': ['linear', 'rbf'],
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__gamma': ['scale', 0.1, 1]
}

grid = GridSearchCV(svr_pipe, param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train_small, y_train_small)
print("Best params:", grid.best_params_)

best = grid.best_estimator_
y_pred = best.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Test RMSE:", rmse)  # ~0.5–0.7
```
