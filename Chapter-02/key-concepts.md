# 🧠 Chapter 2: Key Concepts & Glossary

Welcome to the Key Concepts guide for Chapter 2. This document covers the essential terminology, pipeline structures, and performance benchmarks for building an end-to-end Machine Learning project.

---

## 📖 Glossary (A-Z)

| Term | Definition | Practical Example / Code |
| :--- | :--- | :--- |
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
| :--- | :--- | :--- |
| **Num Pipeline** | Imputer &rarr; Scaler | Clean & scale numeric features |
| **Cat Pipeline** | Imputer &rarr; OneHotEncoder | Convert text categories to numeric binary values |
| **Ratio Pipeline** | Imputer &rarr; Ratio &rarr; Scaler | Create ratio features (e.g., bedrooms per room) |
| **Log Pipeline** | Imputer &rarr; Log &rarr; Scaler | Fix heavy-tailed distributions using logarithmic scale |
| **Geo Pipeline** | RBF &rarr; Clusters | Geographic similarity features based on cluster centroids |
| **Full Pipeline** | All of the above &rarr; Model | Complete preprocessing pipeline integrated with the model |

---

## 📊 Model Performance Summary

| Model | Training RMSE | CV RMSE | Test RMSE | Issue / Status |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | \$68,687 | \$69,858 | — | ⚠️ Underfitting |
| **Decision Tree** | \$0 *(perfect)* | \$66,868 | — | 🚨 Overfitting |
| **Random Forest (default)** | \$17,474 | \$47,019 | — | ⚠️ Moderate Overfitting |
| **Random Forest (tuned)** | **~\$15,000** | **\$44,042** | **\$41,424** |  Best Overall (Selected) |

---

## 🧠 Memory Aid: The 80/20 Rule

> [!TIP]
> **Data Prep vs. Modeling**
> - **80%** of the work is **data preparation** – cleaning, transforming, feature engineering, and pipeline design.
> - **20%** is **model selection and hyperparameter tuning**.
> - *Note: Deployment and monitoring is a continuous, full-time lifecycle (MLOps).*

---

## 📌 Common Pitfalls & Solutions

| Pitfall | Cause / Risk | Solution |
| :--- | :--- | :--- |
| **Sampling Bias** | Non-representative split | Use stratified sampling (`StratifiedShuffleSplit`) |
| **Missing Values** | Missing entries (`NaN`) | Impute with median or most frequent values |
| **Different Scales** | Features with highly varying ranges | Apply feature scaling (e.g., `StandardScaler`) |
| **Categorical Text** | Models cannot process raw text | One-hot encode or ordinal encode categorical columns |
| **Overfitting** | Model is too complex for the training data | Simplify model, apply regularization, or collect more data |
| **Underfitting** | Model is too simple to capture patterns | Use a more complex model, engineer better features |
| **Data Mismatch** | Train and test data from different distributions | Use a train-dev set to diagnose |
| **Tuning on Test Set** | Data snooping bias | **NEVER** tune hyperparameters on the test set; use cross-validation |