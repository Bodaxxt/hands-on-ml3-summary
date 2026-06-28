# 🏠 Chapter 2 – End-to-End Machine Learning Project

This document provides a comprehensive summary of the workflow, key decisions, code implementations, and outcomes from Chapter 2.

## 🎯 Project Overview
- **Goal:** Predict median house values in California districts using census data.
- **Dataset:** California Housing Prices (1990 census) – 20,640 districts, 10 attributes.
- **Approach:** Complete end-to-end Machine Learning workflow from framing the problem to model deployment.

---

## 📋 The ML Project Workflow

| Step | Description |
| :---: | :--- |
| **1** | **Frame the problem:** Define the business objective, select performance measures. |
| **2** | **Get the data:** Download, explore, and split into train/test sets. |
| **3** | **Explore & visualize:** Understand patterns, correlations, and distributions. |
| **4** | **Prepare the data:** Clean, transform, and engineer features. |
| **5** | **Train models:** Try multiple algorithms (Linear Regression, Decision Tree, Random Forest). |
| **6** | **Fine-tune:** Perform hyperparameter tuning (Grid Search, Randomized Search). |
| **7** | **Evaluate:** Test model performance on unseen data (Test Set). |
| **8** | **Launch & monitor:** Deploy, monitor, and establish retraining schedules. |

---

## 📊 Key Steps in Detail

### 1️⃣ Framing the Problem
* **Type:** Supervised learning (labeled dataset).
* **Task:** Regression (predicting a continuous numeric value).
* **Scope:** Multiple regression (utilizing multiple predictive features).
* **Mode:** Batch learning (data easily fits in memory).
* **Performance Measure:** RMSE (Root Mean Square Error).

### 2️⃣ Getting the Data

```python
def load_housing_data():
    # Downloads, extracts, and loads the dataset
    # Returns pandas DataFrame
```

> [!WARNING]
> **Data Snooping Bias:** Always create a test set **BEFORE** you look at the data to prevent your brain from recognizing patterns and biasing model selection.

#### Stratified Sampling

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

> **Why?** Preserves the proportion of income categories in the test set to avoid sampling bias (since income is a critical predictor).

---

### 3️⃣ Data Exploration (EDA)

#### Quick Look at Data Structure
```python
housing.head()
housing.info()      # 20,640 rows, 10 columns, total_bedrooms has 207 nulls
housing.describe()  # Statistical summary
housing.hist(bins=50, figsize=(12, 8))  # Visualize distributions
```

#### Key Observations & Cleaning Requirements

| Attribute | Issue | Proposed Solution |
| :--- | :--- | :--- |
| **`median_income`** | Capped at 15, pre-scaled (not USD) | Understand & document the scale |
| **`median_house_value`** | Capped at \$500,000 | Remove or handle target caps carefully |
| **`total_bedrooms`** | 207 missing values | Impute with median value |
| **`ocean_proximity`** | Categorical text | One-hot encode into binary vectors |
| **All Numeric features** | Different scales | Apply feature scaling (e.g., `StandardScaler`) |

#### Correlation Analysis
```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```

| Feature | Correlation with House Value |
| :--- | :---: |
| **`median_income`** | `0.688` (Strong positive correlation) |
| **`total_rooms`** | `0.137` (Weak positive correlation) |
| **`latitude`** | `-0.140` (Weak negative correlation) |

> [!NOTE]
> `median_income` is by far the strongest predictor of house prices.

---

### 4️⃣ Data Preparation (The Pipeline)

#### Clean Training Set
```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

#### Preprocessing Pipelines

```python
# Numerical Pipeline
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

# Categorical Pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
```

#### Custom Feature Engineering Transformers
- **Ratio Transformer:** Combines features (e.g., `bedrooms/rooms`, `rooms/household`, `people/household`).
- **Log Transformer:** Reduces skewness in heavy-tailed features (e.g., `population`, `total_rooms`).
- **Cluster Similarity:** Computes RBF similarity to geographical cluster centroids.

#### Full Preprocessing Pipeline
```python
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", ClusterSimilarity(n_clusters=10), ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder=num_pipeline)
```

---

### 5️⃣ Model Training & Evaluation

#### Linear Regression
```python
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
lin_rmse = mean_squared_error(housing_labels, lin_reg.predict(housing), squared=False)
# Result: ~68,687 RMSE
```

#### Decision Tree
```python
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
tree_rmse = cross_val_score(tree_reg, housing, housing_labels,
                            scoring="neg_root_mean_squared_error", cv=10)
# Result: ~66,868 RMSE (but 0 on training set -> Overfitting!)
```

#### Random Forest
```python
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmse = cross_val_score(forest_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
# Result: ~47,019 RMSE -> Significant improvement
```

#### Model Comparison Summary

| Model | CV RMSE | Main Issue / Status |
| :--- | :---: | :--- |
| **Linear Regression** | \$69,858 | Underfitting |
| **Decision Tree** | \$66,868 | Overfitting (0 Training RMSE) |
| **Random Forest** | \$47,019 | Moderate overfitting, but best baseline |

---

### 6️⃣ Fine-Tuning (Hyperparameter Optimization)

#### Grid Search
```python
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```
- **Best Parameters:** `n_clusters=15`, `max_features=6`
- **Best RMSE:** **\$44,042** (improved from \$47,019)

#### Randomized Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'preprocessing__geo__n_clusters': randint(3, 50),
    'random_forest__max_features': randint(2, 20),
}

rnd_search = RandomizedSearchCV(full_pipeline, param_distribs, n_iter=10, cv=3)
rnd_search.fit(housing, housing_labels)
```
> [!TIP]
> Better for broad hyperparameter exploration. Runs in a fixed iteration budget and scales well.

---

### 7️⃣ Feature Importance Analysis
```python
feature_importances = final_model["random_forest"].feature_importances_
sorted(zip(feature_importances, feature_names), reverse=True)
```
* **Top Features:**
  1. `log_median_income` (18.7%)
  2. `ocean_proximity_INLAND` (7.5%)
  3. `bedrooms_ratio` (6.9%)

---

### 8️⃣ Final Evaluation on Test Set
```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)  # ~41,424
```
* **95% Confidence Interval:** `[39,275 – 43,467]`

---

### 9️⃣ Launch, Monitor, Maintain

#### Save Model
```python
import joblib
joblib.dump(final_model, "california_housing_model.pkl")
```

#### Load Model
```python
final_model_reloaded = joblib.load("california_housing_model.pkl")
predictions = final_model_reloaded.predict(new_data)
```

#### MLOps Essentials

| Task | Description |
| :--- | :--- |
| **Monitoring** | Track model performance metrics over time and verify quality of incoming data. |
| **Retraining** | Set up pipelines to automatically retrain the model on fresh data periodically. |
| **Backups** | Archive and version every model artifact and training dataset split. |
| **Rollback** | Enable quick rollbacks to the last stable model in production if issues occur. |

---

## 🧠 Key Takeaways

* 📊 **Stratified Sampling** is crucial to prevent sampling bias on small datasets.
* ⚙️ **Data Preparation** is **80%** of the work—invest in modular and robust scikit-learn pipelines.
* 🛠️ **Feature Engineering** (log transformation, ratio attributes, clustering) yields the biggest boosts in performance.
* 🔬 **Cross-Validation** is the only way to get honest, reliable estimates of model performance.
* 🤝 **Ensemble Methods** (like Random Forest) typically outperform single estimator models by combining their predictions.

---

## 💡 Golden Rules

1. > *"Garbage in, garbage out" — quality data and feature engineering beat model complexity every time.*
2. > *"Never touch the test set until the final evaluation" — prevent data snooping bias.*
3. > *"Simple models that generalize well are always better than complex models that memorize noise."*

---

## 🚀 Next Steps

- [ ] Try training other regression models (e.g., SVR, XGBoost, or deep Neural Networks).
- [ ] Experiment with different feature combinations and custom transformers.
- [ ] Deploy the saved model as a REST API or inside a serverless cloud function.
- [ ] Test the pipeline on Kaggle California Housing competition entries.