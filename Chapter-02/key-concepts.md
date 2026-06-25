
---

## 📁 Chapter-02/key-concepts.md

```markdown
# Key Concepts – Chapter 2

## 📖 Glossary (A-Z)

| Term | Definition | Example |
|------|------------|---------|
| **Batch Learning** | Train once, deploy, retrain periodically | California housing model retrained monthly |
| **Categorical Attribute** | Text values representing categories | `ocean_proximity` |
| **ColumnTransformer** | Applies different transformers to different columns | Numerical vs Categorical pipelines |
| **Cross-Validation (CV)** | Evaluate model on multiple folds of training data | 10-fold CV |
| **Data Leakage** | Info from test set leaks into training | Using test set for tuning |
| **Data Snooping Bias** | Peeking at test set influences model choice | Selecting model because it works on test set |
| **EDA** | Exploratory Data Analysis – understand data before modeling | `info()`, `describe()`, `hist()` |
| **Ensemble** | Combine multiple models for better performance | Random Forest |
| **Feature Engineering** | Creating new features from existing ones | `rooms_per_household`, `log_population` |
| **Feature Importance** | Measures how much each feature contributes to predictions | `feature_importances_` |
| **Feature Scaling** | Normalize features to similar ranges | `StandardScaler`, `MinMaxScaler` |
| **Grid Search** | Exhaustive search over hyperparameter combinations | `GridSearchCV` |
| **Hyperparameter** | Settings set before training (not learned) | `n_clusters`, `max_features` |
| **Imputation** | Filling missing values | `SimpleImputer` |
| **Inference** | Making predictions on new data | `model.predict()` |
| **MLOps** | ML Operations – deployment, monitoring, maintenance | Joblib, retraining, monitoring |
| **One-Hot Encoding** | Convert categories to binary columns | `OneHotEncoder` |
| **Pipeline** | Chain of transformations | `make_pipeline(Imputer, Scaler)` |
| **Randomized Search** | Random hyperparameter combinations | `RandomizedSearchCV` |
| **RMSE** | Root Mean Square Error – regression performance | `mean_squared_error(squared=False)` |
| **Sampling Bias** | Non-representative sample | Literary Digest poll 1936 |
| **Stratified Sampling** | Preserve category proportions in split | `StratifiedShuffleSplit` |
| **Test Set** | Unseen data for final evaluation | `strat_test_set` |
| **Train-Dev Set** | For detecting data mismatch | Web vs mobile images |
| **Underfitting** | Model too simple | Linear Regression on housing |
| **Overfitting** | Model memorizes training data | Decision Tree (0 error on train) |
| **Validation Set** | For tuning and model selection | Part of training set used for CV |

---

## 🔄 Pipeline Comparison

| Pipeline | Components | Purpose |
|----------|------------|---------|
| **Num Pipeline** | Imputer → Scaler | Clean & scale numeric features |
| **Cat Pipeline** | Imputer → OneHotEncoder | Convert text to numbers |
| **Ratio Pipeline** | Imputer → Ratio → Scaler | Create ratio features |
| **Log Pipeline** | Imputer → Log → Scaler | Fix heavy-tailed distributions |
| **Geo Pipeline** | RBF → Clusters | Geographic similarity features |
| **Full Pipeline** | All of the above → Model | Complete preprocessing + training |

---

## 📊 Model Performance Summary

| Model | Training RMSE | CV RMSE | Test RMSE | Issue |
|-------|---------------|---------|-----------|-------|
| Linear Regression | 68,687 | 69,858 | - | Underfitting |
| Decision Tree | 0 (perfect) | 66,868 | - | Overfitting |
| Random Forest (default) | 17,474 | 47,019 | - | Some overfitting |
| Random Forest (tuned) | ~15,000 | 44,042 | 41,424 | Best overall |

---

## 🧠 Memory Aid: The 80/20 Rule

> **80%** of the work is **data preparation** – cleaning, transforming, feature engineering.

> **20%** is **model selection and tuning**.

> **Deployment and monitoring** is a full-time job on its own.

---

## 📌 Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| **Sampling bias** | Use stratified sampling |
| **Missing values** | Impute with median or most frequent |
| **Different scales** | Use StandardScaler |
| **Categorical text** | One-hot encode or ordinal encode |
| **Overfitting** | Simplify model, use regularization, or get more data |
| **Underfitting** | More complex model, better features |
| **Data mismatch** | Use train-dev set |
| **Tuning on test set** | NEVER do this – use validation set |