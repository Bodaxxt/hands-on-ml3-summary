# 🧠 Chapter 3: Key Concepts & Glossary

Welcome to the Key Concepts guide for Chapter 3. This document covers essential terminology, metric comparisons, classification types, and practical mnemonics for classification models.

---

## 📖 Glossary (A-Z)

| Term | Definition | Example / Practical Use |
| :--- | :--- | :--- |
| **Accuracy** | Ratio of correct predictions | `(TP + TN) / Total` |
| **AUC** | Area Under ROC Curve | Higher = better performance |
| **Binary Classification** | Two classes only | `5` vs `non-5` |
| **Confusion Matrix** | Table of correct/incorrect predictions | `TP`, `FP`, `FN`, `TN` grid |
| **Data Augmentation** | Artificial increase of training data | Shift, rotate, add noise |
| **Decision Threshold** | Cutoff for classification | If `score > threshold` &rarr; Positive |
| **F1 Score** | Harmonic mean of Precision & Recall | `2 * (P * R) / (P + R)` |
| **FPR** | False Positive Rate | `FP / (FP + TN)` |
| **Macro Average** | Average metric per class (equal weight) | Used in multilabel evaluation |
| **Multiclass** | More than 2 classes | Digits `0`-`9` |
| **Multilabel** | Multiple True/False labels per instance | `Alice` + `Bob` in one image |
| **Multioutput** | Multiple labels with >2 values | Pixel intensities (`0`-`255`) |
| **OvO** | One-vs-One | `N * (N - 1) / 2` classifiers |
| **OvR** | One-vs-Rest | `N` classifiers |
| **Precision** | Accuracy of positive predictions | `TP / (TP + FP)` |
| **Recall** | Coverage of positive class | `TP / (TP + FN)` |
| **ROC Curve** | TPR vs FPR | Visual performance metric |
| **Specificity** | True Negative Rate | `TN / (FP + TN)` |
| **TPR** | True Positive Rate | Same as Recall |
| **Weighted Average** | Average metric weighted by class support | Used in multilabel evaluation |

---

## 📊 Metrics Quick Reference

| Metric | Formula | Use Case |
| :--- | :--- | :--- |
| **Precision** | `TP / (TP + FP)` | When false positives are costly (e.g., spam filter) |
| **Recall** | `TP / (TP + FN)` | When false negatives are costly (e.g., medical diagnosis) |
| **F1** | `2 * (P * R) / (P + R)` | When you need a balance |
| **Accuracy** | `(TP + TN) / Total` | Only when classes are balanced |
| **ROC AUC** | Area under TPR vs FPR | Overall model quality |

---

## 🧠 Comparison: OvR vs OvO

| Feature | OvR (One-vs-Rest) | OvO (One-vs-One) |
| :--- | :--- | :--- |
| **# Classifiers** | `N` | `N * (N - 1) / 2` |
| **Speed** | Faster | Slower |
| **Training Data** | All data | Two classes only |
| **Best for** | Most algos | SVC (slow algos) |

---

## 🔄 Classification Types

```text
Classification
│
├── Binary (2 classes)
│   └── Example: 5 vs non-5
│
├── Multiclass (>2 classes)
│   ├── Strategies: OvR, OvO
│   └── Example: Digits 0-9
│
├── Multilabel (multiple binary labels)
│   └── Example: Alice + Bob in image
│
└── Multioutput (multiple multi-class labels)
    └── Example: Denoising images (pixels 0-255)
```

---

## 📝 Mnemonics

| Concept | Memory Aid |
| :--- | :--- |
| **Precision** | *"Of all I said Yes, how many were right?"* |
| **Recall** | *"Of all true Yes, how many did I catch?"* |
| **F1** | *"Balance between P and R"* |
| **ROC** | *"Trade-off between catching positives and avoiding false alarms"* |
| **OvR** | *"One vs Rest"* |
| **OvO** | *"One vs One"* |

---

> [!TIP]
> *"Choose your metric wisely – it defines what your model optimizes for."*