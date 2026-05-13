# Solutions – Chapter 1 Exercises

## 📝 Question & Answer

### 1. How would you define machine learning?
Machine learning is the science (and art) of programming computers to learn from data, rather than following explicit rules.

---

### 2. Can you name four types of applications where it shines?
1. **Image classification** (product sorting, tumor detection)
2. **Natural language processing** (news classification, chatbots)
3. **Speech recognition** (voice commands)
4. **Anomaly detection** (credit card fraud)

---

### 3. What is a labeled training set?
A dataset where each instance has both input features AND the correct output (label). Example: emails with "spam" or "ham" labels.

---

### 4. What are the two most common supervised tasks?
- **Classification** (predict a category/discrete class)
- **Regression** (predict a numeric value)

---

### 5. Can you name four common unsupervised tasks?
1. **Clustering** (grouping similar instances)
2. **Visualization** (plotting high-dimensional data in 2D/3D)
3. **Dimensionality reduction** (reducing number of features)
4. **Anomaly detection** (finding unusual instances)

---

### 6. What type of algorithm would you use to allow a robot to walk in various unknown terrains?
**Reinforcement Learning** – the robot learns by trial and error, receiving rewards for good actions.

---

### 7. What type of algorithm would you use to segment your customers into multiple groups?
**Clustering algorithm** (unsupervised learning) – it finds natural groups without labels.

---

### 8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
**Supervised learning** – you have historical emails with labels (spam/ham).

---

### 9. What is an online learning system?
A system that learns incrementally by feeding it data instances sequentially (individually or in mini-batches), adapting quickly to change.

---

### 10. What is out-of-core learning?
Training on datasets too large to fit in memory by loading parts of the data, training on them, and repeating until all data is processed.

---

### 11. What type of algorithm relies on a similarity measure to make predictions?
**Instance-based learning** (e.g., K-Nearest Neighbors)

---

### 12. What is the difference between a model parameter and a model hyperparameter?
- **Parameter:** Learned from training data (weights, biases)
- **Hyperparameter:** Set before training (learning rate, k in KNN)

---

### 13. What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
- **Search for:** Optimal model parameters (θ)
- **Strategy:** Minimize a cost function (e.g., MSE)
- **Predictions:** Apply the learned equation to new instances

---

### 14. Can you name four of the main challenges in machine learning?
1. Insufficient quantity of training data
2. Nonrepresentative training data
3. Overfitting
4. Underfitting

---

### 15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
**This is overfitting.**

Solutions:
1. Simplify the model (fewer parameters)
2. Apply regularization
3. Get more training data or reduce noise

---

### 16. What is a test set, and why would you want to use it?
A test set is data never used during training, used once at the end to estimate the model's generalization error.

---

### 17. What is the purpose of a validation set?
To evaluate multiple candidate models, tune hyperparameters, and select the best model before final testing.

---

### 18. What is the train-dev set, when do you need it, and how do you use it?
- **What:** A subset of training data held out to detect data mismatch
- **When:** When training data differs from production data
- **How:** If model performs well on train-dev but poorly on dev → data mismatch

---

### 19. What can go wrong if you tune hyperparameters using the test set?
You'll get an overly optimistic estimate of generalization error. The model likely won't perform as well on new data because it's been indirectly "fitted" to the test set.

---

## 📊 Summary Table

| # | Concept | Answer |
|---|---------|--------|
| 1 | ML Definition | Computers learning from data |
| 2 | 4 applications | Images, text, speech, fraud |
| 3 | Labeled set | Data with correct answers |
| 4 | Supervised tasks | Classification, Regression |
| 5 | Unsupervised tasks | Clustering, Viz, Dim red, Anomaly |
| 6 | Robot walking | Reinforcement Learning |
| 7 | Customer segmentation | Clustering |
| 8 | Spam detection | Supervised Learning |
| 9 | Online learning | Incremental training |
| 10 | Out-of-core | Train on data > RAM |
| 11 | Similarity measure | Instance-based learning |
| 12 | Parameter vs Hyperparameter | Learned vs set before training |
| 13 | Model-based search | Minimize cost function |
| 14 | 4 challenges | Insufficient, nonrepresentative, overfitting, underfitting |
| 15 | Overfitting | Simplify, regularize, more data |
| 16 | Test set | Final evaluation (once) |
| 17 | Validation set | Tuning & model selection |
| 18 | Train-dev set | Detect data mismatch |
| 19 | Tuning with test set | Overly optimistic results |