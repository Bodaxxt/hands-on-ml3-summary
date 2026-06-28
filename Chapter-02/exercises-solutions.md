# 🛠️ Solutions – Chapter 2 Exercises

This document contains full solution code and brief explanations for the exercises in Chapter 2 of the machine learning workbook.

---

## 🧩 Exercise 1: SVR Regressor

```python
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Use only first 5000 samples for speed
X_train_small = X_train[:5000]
y_train_small = y_train[:5000]

# Linear kernel
svr_linear = make_pipeline(preprocessing, SVR(kernel="linear", C=1.0))
scores_linear = -cross_val_score(svr_linear, X_train_small, y_train_small,
                                 scoring="neg_root_mean_squared_error", cv=3)

# RBF kernel
svr_rbf = make_pipeline(preprocessing, SVR(kernel="rbf", C=1.0, gamma="scale"))
scores_rbf = -cross_val_score(svr_rbf, X_train_small, y_train_small,
                              scoring="neg_root_mean_squared_error", cv=3)

print(f"SVR Linear RMSE: {scores_linear.mean():.2f}")
print(f"SVR RBF RMSE: {scores_rbf.mean():.2f}")
```

> [!NOTE]
> **Expected Outcome:** The RBF kernel usually performs significantly better (~55,000–60,000 RMSE) compared to the linear kernel in non-linear datasets.

---

## 🧩 Exercise 2: Replace GridSearch with RandomizedSearch

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'random_forest__n_estimators': randint(50, 200),
    'random_forest__max_depth': randint(5, 30),
    'random_forest__min_samples_split': randint(2, 10),
    'random_forest__min_samples_leaf': randint(1, 5),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

rnd_search.fit(housing, housing_labels)
print(rnd_search.best_params_)
```

> [!TIP]
> **Why use RandomizedSearch?** When the hyperparameter search space is large, `RandomizedSearchCV` is much faster and more resource-efficient than an exhaustive `GridSearchCV`.

---

## 🧩 Exercise 3: Add SelectFromModel

```python
from sklearn.feature_selection import SelectFromModel

selection_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("selector", SelectFromModel(RandomForestRegressor(random_state=42), threshold="median")),
    ("model", RandomForestRegressor(random_state=42))
])

selection_pipeline.fit(housing, housing_labels)
scores = -cross_val_score(selection_pipeline, housing, housing_labels,
                          scoring="neg_root_mean_squared_error", cv=3)
print(f"RMSE after selection: {scores.mean():.2f}")
```

> [!NOTE]
> **Effect:** Drops the least important features according to the feature selector, which reduces model dimensionality and can improve training speed or generalize better on unseen data.

---

## 🧩 Exercise 4: Custom KNN Transformer

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class KNNFeature(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.knn.fit(X, y)
        return self

    def transform(self, X):
        return self.knn.predict(X).reshape(-1, 1)

# Add to pipeline
preprocessing_with_knn = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
    ("geo_knn", KNNFeature(n_neighbors=5), ["latitude", "longitude"]),
], remainder=num_pipeline)
```

> [!NOTE]
> **Core Idea:** Adds the predicted price of the nearest geographical neighbors as a new numeric feature for the regressor, utilizing spatial correlation.

---

## 🧩 Exercise 5: Explore Preparation Options with GridSearch

```python
param_grid = [
    {
        'preprocessing__geo__n_clusters': [5, 10, 15],
        'preprocessing__log__function_transformer__func': [np.log, np.sqrt],
        'model__n_estimators': [50, 100, 150],
    }
]

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error'
)

grid_search.fit(housing, housing_labels)
print(grid_search.best_params_)
```

> [!NOTE]
> **Goal:** Search over both preprocessing choices (e.g., Log vs. Square Root transformation) and model configurations (e.g., number of estimators) simultaneously.

---

## 🧩 Exercise 6: Implement StandardScalerClone

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.with_mean:
            X_copy -= self.mean_
        return X_copy / self.scale_

    def inverse_transform(self, X):
        X_copy = X * self.scale_
        if self.with_mean:
            X_copy += self.mean_
        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Number of input features doesn't match")
            if hasattr(self, "feature_names_in_"):
                if not np.array_equal(input_features, self.feature_names_in_):
                    raise ValueError("Input features don't match")
            return np.array(input_features)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.array([f"x{i}" for i in range(self.n_features_in_)])
```

### ⚙️ Main Methods:
- **`fit()`**: Learns the mean and standard deviation of each feature.
- **`transform()`**: Standardizes the features using the learned mean and standard deviation.
- **`inverse_transform()`**: Rescales the features back to their original dimensions.
- **`get_feature_names_out()`**: Helper to track output column names within scikit-learn Pipelines.

---

## ✅ Summary of Solutions

| Exercise | Skill Learned |
| :---: | :--- |
| **1** | Trying different models (SVR) |
| **2** | Using `RandomizedSearch` instead of `GridSearch` |
| **3** | Feature selection with `SelectFromModel` |
| **4** | Building custom transformer using KNN |
| **5** | Searching over preprocessing choices |
| **6** | Implementing `StandardScaler` from scratch |