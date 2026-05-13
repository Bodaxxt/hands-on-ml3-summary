# Chapter 1 – The Machine Learning Tsunami

## 🎯 What is Machine Learning?
> Programming computers to learn from data, instead of writing explicit rules.

**Arthur Samuel (1959):** "Field of study that gives computers the ability to learn without being explicitly programmed."

**Tom Mitchell (1997):** "A program learns from experience E with respect to task T and performance measure P, if its performance on T, as measured by P, improves with experience E."

---

## 🧠 Types of ML Systems

### By Supervision
| Type | What it does | Examples |
|------|--------------|----------|
| **Supervised** | Learn from labeled data | Classification, Regression |
| **Unsupervised** | Find patterns in unlabeled data | Clustering, Visualization |
| **Self-Supervised** | Generate labels from data itself | Masked autoencoders |
| **Reinforcement** | Learn by rewards/punishments | Game AI, Robotics |

### By Learning Method
- **Batch Learning:** Train once, then deploy (retrain periodically)
- **Online Learning:** Learn incrementally as data arrives

### By Generalization
- **Instance-Based:** Memorize examples, compare new ones (KNN)
- **Model-Based:** Build a model, make predictions (Linear Regression)

---

## 📊 The ML Workflow

1. **Study the data** – understand features, relationships
2. **Select a model** – choose algorithm (linear, polynomial, etc.)
3. **Train the model** – find best parameters (minimize cost function)
4. **Make predictions (inference)** – apply to new data

---

## ⚠️ Main Challenges

### Bad Data
| Problem | Description | Solution |
|---------|-------------|----------|
| **Insufficient data** | Need thousands/millions of examples | More data, augmentation |
| **Nonrepresentative** | Sample doesn't reflect reality | Stratified sampling |
| **Poor quality** | Errors, outliers, noise | Clean data |
| **Irrelevant features** | Useless or redundant features | Feature engineering |

### Bad Model
| Problem | Description | Solution |
|---------|-------------|----------|
| **Overfitting** | Too complex, memorizes noise | Regularization, more data, simplify |
| **Underfitting** | Too simple, misses patterns | Complex model, better features |

---

## 🔧 Regularization
> Constraining a model to make it simpler and reduce overfitting

**Degrees of freedom:** Number of parameters the model can tweak (θ₀, θ₁, θ₂...)

**Balance:** Fit training data well + keep model simple

---

## 📐 Testing & Validation

| Set | Purpose | How often |
|-----|---------|-----------|
| **Training** | Learn parameters | Every epoch |
| **Validation** | Tune hyperparameters, select model | Many times |
| **Test** | Final evaluation (generalization error) | Once at the end |
| **Train-Dev** | Detect data mismatch | After training |

### Data Mismatch
- When training data (easy to get) ≠ production data
- Solution: Keep dev/test representative of production

---

## 🏷️ Key Terms

| Term | Meaning |
|------|---------|
| **Parameter** | Learned from data (θ₀, weights, biases) |
| **Hyperparameter** | Set before training (k in KNN, λ in regularization) |
| **Label** | Output in classification (spam/ham) |
| **Target** | Output in regression (price) |
| **Feature** | Input (mileage, age, brand) |
| **Inference** | Making predictions on new data |
| **Generalization** | Performing well on unseen data |

---

## 📈 Learning Curves
- **Overfitting:** Large gap between train & validation performance
- **Underfitting:** Both train & validation performance are poor

---

## 🧪 Exercises Summary

| # | Question | Answer |
|---|----------|--------|
| 1 | Define ML | Computers learning from data |
| 2 | 4 applications | Images, text, speech, fraud detection |
| 3 | Labeled training set | Data with correct answers |
| 4 | 2 supervised tasks | Classification, Regression |
| 5 | 4 unsupervised tasks | Clustering, Viz, Dim reduction, Anomaly detection |
| 6 | Robot walking | Reinforcement learning |
| 7 | Customer segmentation | Clustering |
| 8 | Spam detection | Supervised learning |
| 9 | Online learning | Incremental training |
| 10 | Out-of-core | Training on data larger than RAM |
| 11 | Similarity measure | Instance-based learning |
| 12 | Parameter vs Hyperparameter | Learned vs set by you |
| 13 | Model-based search | Minimize cost function |
| 14 | 4 challenges | Insufficient data, nonrepresentative, overfitting, underfitting |
| 15 | Overfitting | Simplify, regularize, more data |
| 16 | Test set | Final evaluation (once) |
| 17 | Validation set | Tuning & model selection |
| 18 | Train-dev set | Detect data mismatch |
| 19 | Tuning with test set | Overly optimistic results |

---

## 🚀 My Key Takeaways

1. **"Garbage in, garbage out"** – always validate your data
2. **Simplicity generalizes better** – prefer simpler models
3. **Never touch the test set until the end** – or you'll cheat yourself
4. **Data mismatch is real** – dev/test must represent production
5. **Regularization is your friend** – it's the difference between memorizing and learning