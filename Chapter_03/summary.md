# 🏷️ Chapter 3 – Classification

## 📘 Overview

This chapter covers **classification** – one of the most common supervised learning tasks. We explore:
- Binary classification (5 vs non-5)
- Performance metrics (Precision, Recall, F1, ROC, AUC)
- Multiclass classification (OvR, OvO)
- Multilabel and Multioutput classification
- Error analysis and data augmentation

---

## 🧠 Key Concepts

### Binary Classification
- Classifier distinguishes between two classes.
- Example: `5` vs `not 5` using `SGDClassifier`.

### Performance Metrics
| Metric | Definition |
| :--- | :--- |
| **Accuracy** | `(TP + TN) / Total` |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **ROC Curve** | Plot of TPR vs FPR |
| **AUC** | Area Under ROC Curve |

### Precision/Recall Trade-off
- Increasing Precision &rarr; decreases Recall
- Increasing Recall &rarr; decreases Precision
- Controlled by **decision threshold**

### Confusion Matrix
- Shows correct & incorrect predictions per class.
- Helps identify which classes are confused.

### Multiclass Classification
- More than two classes (e.g., digits `0`-`9`).
- Strategies:
  - **OvR (One-vs-Rest)**: `N` classifiers (one per class)
  - **OvO (One-vs-One)**: `N * (N - 1) / 2` classifiers

### Multilabel Classification
- Each instance can have multiple labels (e.g., `Alice` + `Bob` in one image).
- Evaluated using F1 (macro or weighted average).

### Multioutput Classification
- Generalization of multilabel where each label has more than 2 values.
- Example: Denoising images (predict pixel intensities `0`-`255`).

### Error Analysis
- Use confusion matrix to find weaknesses.
- Check individual errors to understand model behavior.

### Data Augmentation
- Artificially increase training set by shifting, rotating, or adding noise.
- Improves generalization.

---

## 💻 Code Examples

### Load MNIST
```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
```

### Train Binary Classifier (5 vs non-5)
```python
from sklearn.linear_model import SGDClassifier

y_train_5 = (y_train == '5')
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

### Confusion Matrix
```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
```

### Precision, Recall, F1
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)
```

### ROC Curve
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_auc_score(y_train_5, y_scores)
```

### Multiclass with SVC (OvO)
```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
```

### Multiclass with SGD (OvR)
```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
```

### Multilabel Classification
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

### Multioutput (Denoising)
```python
import numpy as np

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
knn_clf.fit(X_train_mod, X_train)
clean_digit = knn_clf.predict([X_test_mod[0]])
```

---

## 📊 Summary Table

| Concept | Key Takeaway |
| :--- | :--- |
| **Binary Classification** | Two classes (5 vs non-5) |
| **Precision/Recall** | Trade-off controlled by threshold |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **ROC/AUC** | TPR vs FPR; AUC = performance |
| **Multiclass** | OvR (faster) or OvO (more accurate for SVC) |
| **Multilabel** | Multiple True/False labels per instance |
| **Multioutput** | Multiple labels with >2 possible values |
| **Data Augmentation** | Shift, rotate, or add noise to training data |

---

## ✅ Key Takeaways

- 📊 **Accuracy can be misleading** – use Precision/Recall/F1-score for imbalanced datasets.
- 🎯 **Confusion Matrix** is essential for error analysis.
- ⚖️ **Precision/Recall trade-off** depends on the business problem.
- 🔄 **Multiclass** can be handled via OvR or OvO.
- 🏷️ **Multilabel** requires special metrics (F1 macro/weighted).
- 💡 **Multioutput** is useful for tasks like denoising.
- 🚀 **Data Augmentation** improves generalization.
- 📏 **Scaling inputs** significantly improves SGD performance.

---

> [!TIP]
> *"Classification is the heart of many ML applications. Master the metrics, understand the trade-offs, and always analyze your errors."*