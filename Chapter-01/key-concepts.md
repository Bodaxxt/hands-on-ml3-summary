# Key Concepts – Chapter 1

##  Glossary (A-Z)

| Term | Definition | Example |
|------|------------|---------|
| **Batch Learning** | Train once, deploy, retrain periodically | Spam filter updated weekly |
| **Bias** | Error from wrong assumptions | Assuming linear when data is quadratic |
| **Classification** | Predict discrete class | Spam vs Ham |
| **Clustering** | Group similar instances | Customer segmentation |
| **Cost Function** | Measures model error | MSE (Mean Squared Error) |
| **Data Augmentation** | Artificially increase dataset size | Rotating images |
| **Data Mismatch** | Train data ≠ Production data | Web pics vs Mobile pics |
| **Degrees of Freedom** | Number of adjustable parameters | θ₀ and θ₁ |
| **Dimensionality Reduction** | Reduce number of features | PCA |
| **Feature** | Input variable | Mileage, age, brand |
| **Feature Engineering** | Creating/selecting good features | Combining mileage + age → wear & tear |
| **Generalization** | Performing well on unseen data | The ultimate goal of ML |
| **Hyperparameter** | Set before training | Learning rate, k in KNN |
| **Inference** | Making predictions | model.predict() |
| **Instance-Based Learning** | Memorize & compare | K-Nearest Neighbors |
| **Label** | Output in classification | "spam" |
| **Model-Based Learning** | Build model from data | Linear Regression |
| **Nonrepresentative Data** | Sample doesn't reflect reality | Poll only landline users |
| **Online Learning** | Learn incrementally | Stock price prediction |
| **Out-of-Core Learning** | Train on data larger than RAM | Using batches |
| **Overfitting** | Memorizing noise | High-degree polynomial |
| **Parameter** | Learned from data | θ₀, θ₁ |
| **Regression** | Predict numeric value | Price, temperature |
| **Regularization** | Constrain model to avoid overfitting | Ridge, Lasso |
| **Reinforcement Learning** | Learn by rewards/punishments | Game AI |
| **Sampling Bias** | Flawed collection method | Phone directory only |
| **Self-Supervised Learning** | Generate labels from data | Masked autoencoders |
| **Supervised Learning** | Learn from labeled data | Spam filter |
| **Target** | Output in regression | 350,000 (price) |
| **Test Set** | Final evaluation (once) | Unseen data |
| **Train-Dev Set** | Detect data mismatch | Held-out training data |
| **Underfitting** | Too simple to learn pattern | Linear on quadratic data |
| **Unsupervised Learning** | Find patterns in unlabeled data | Clustering |
| **Validation Set** | Tune hyperparameters | Practice exam |
| **Variance** | Sensitivity to small changes in training data | High-degree polynomial |

---

##  Quick Comparison Tables

### Supervised vs Unsupervised vs RL

| | Supervised | Unsupervised | RL |
|--|------------|--------------|-----|
| **Labels** | Yes | No | Rewards |
| **Goal** | Predict output | Find patterns | Maximize reward |
| **Examples** | Spam filter | Clustering | Game AI |

### Batch vs Online Learning

| | Batch | Online |
|--|-------|--------|
| **Training** | Once | Continuously |
| **Adaptation** | Slow (retrain) | Fast |
| **Resources** | High | Low |
| **Use case** | Stable data | Changing data |

### Instance-Based vs Model-Based

| | Instance-Based | Model-Based |
|--|----------------|--------------|
| **Method** | Memorize & compare | Build model |
| **Prediction** | Similarity measure | Apply equation |
| **Example** | KNN | Linear Regression |

### Overfitting vs Underfitting

| | Overfitting | Underfitting |
|--|-------------|---------------|
| **Train performance** | Excellent | Poor |
| **Test performance** | Poor | Poor |
| **Gap** | Large | Small |
| **Cause** | Too complex | Too simple |
| **Fix** | Regularize, more data | Complex model, better features |

---

##  Memory Aids

**"GIGO"** – Garbage In, Garbage Out (bad data → bad model)

**"TEST"** – Test set = End of Story (use once at the very end)

**"SPORT"** – Supervised = Pattern Observed, Regression Target (numeric)

**"CURD"** – Common Unsupervised = Clustering, Visualization, Dim Reduction, Anomaly detection

**"VIT"** – Validation = Iterative Tuning (use many times)

**"PARAM"** – Parameter = Automatically learned, Hyperparameter = Manually set