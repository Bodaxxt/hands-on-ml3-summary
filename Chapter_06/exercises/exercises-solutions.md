# 📝 Exercises & Detailed Solutions – Chapter 6

> **Chapter 6: Decision Trees**  
> Questions, conceptual reviews, and practical exercise walkthroughs.

---

## ❓ Review Questions & Answers

### Q1: What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with 1 million instances?
**Answer:**
A well-balanced binary tree with $m$ leaf nodes has a depth of $\approx \log_2(m)$.
For $m = 10^6$ instances:
$$\text{depth} \approx \log_2(10^6) \approx 19.93 \approx 20$$
If the tree is not completely balanced (e.g. some branches stop earlier or are unbalanced), the depth will generally be slightly higher (between 20 and 30), or up to $m-1$ in the worst-case degenerate scenario.

---

### Q2: Is a node's Gini impurity generally lower or greater than its parent's? Is it *generally* lower/greater, or *always* lower/greater?
**Answer:**
- A child node's Gini impurity is **generally lower** than its parent's because the CART algorithm minimizes the weighted average impurity of its children:
  $$J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$
- However, it is **not always lower** for every individual child! One child might have a higher Gini impurity than the parent as long as the other child is very pure and the overall *weighted average* impurity decreases.

---

### Q3: If a Decision Tree is overfitting the training set, is it a good idea to try decreasing `max_depth`?
**Answer:**
**Yes.** Decreasing `max_depth` restricts the model's capacity, forcing it to produce shallower trees with fewer splits. This adds regularization and reduces overfitting (reducing variance).

---

### Q4: If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?
**Answer:**
**No.** Decision Trees are invariant to monotonic transformations and feature scaling (e.g., standardizing or MinMax scaling). Feature scale does not affect splitting thresholds at all. To address underfitting:
- Increase `max_depth` or remove constraints.
- Decrease `min_samples_split` / `min_samples_leaf`.
- Engineer more predictive features.

---

### Q5: If it takes 1 hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree on a training set containing 10 million instances?
**Answer:**
The computational complexity of training a Decision Tree is:
$$\mathcal{O}(n \times m \log_2(m))$$
If $m_1 = 10^6$ takes $t_1 = 1\text{ hour}$, then for $m_2 = 10 \times 10^6$:
$$\text{Ratio} = \frac{m_2 \log_2(m_2)}{m_1 \log_2(m_1)} = \frac{10 \cdot m_1 \cdot \log_2(10 \cdot m_1)}{m_1 \log_2(m_1)} = 10 \times \frac{\log_2(10^7)}{\log_2(10^6)} = 10 \times \frac{\approx 23.25}{\approx 19.93} \approx 11.7$$
Thus, it will take approximately **11.7 hours** (roughly 11 to 12 hours).

---

### Q6: If your training set contains 100,000 instances, will setting `presort=True` speed up training?
**Answer:**
**No.** Presorting data (`presort=True`) only speeds up training for small datasets (a few thousand instances). For large datasets (such as 100,000+ instances), presorting significantly slows down training and consumes extensive memory. (Note: in modern Scikit-Learn versions, the `presort` parameter was deprecated and removed for this reason).

---

## 💻 Practical Coding Exercises

### Exercise 7: Train and Fine-Tune a Decision Tree for the Moons Dataset
```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Generate moons dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Use GridSearchCV to find the best hyperparameters
params = {
    'max_leaf_nodes': list(range(2, 100)),
    'min_samples_split': [2, 3, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# 4. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%") # Should be > 85%
```

---

### Exercise 8: Grow a Random Forest from Individual Trees
```python
from sklearn.model_selection import ShuffleSplit
from scipy.stats import mode

# 1. Generate 1,000 subsets of training data (100 instances each)
n_trees = 1000
n_instances = 100
mini_subsets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, _ in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_subsets.append((X_mini_train, y_mini_train))

# 2. Train 1,000 trees on mini subsets
from sklearn.base import clone
forest = [clone(grid_search.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_subsets):
    tree.fit(X_mini_train, y_mini_train)
    accuracy_scores.append(tree.score(X_test, y_test))

print(f"Average individual tree accuracy: {np.mean(accuracy_scores) * 100:.2f}%")

# 3. Majority-vote ensemble prediction
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

y_pred_votes, n_votes = mode(Y_pred, axis=0)
ensemble_acc = accuracy_score(y_test, y_pred_votes.reshape([-1]))
print(f"Forest Ensemble Accuracy: {ensemble_acc * 100:.2f}%") # Notice the significant boost!
```
