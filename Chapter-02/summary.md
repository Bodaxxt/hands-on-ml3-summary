# Chapter 2 – End-to-End Machine Learning Project

## 🎯 Project Overview
**Goal:** Predict median house values in California districts using census data.

**Dataset:** California Housing Prices (1990 census) – 20,640 districts, 10 attributes.

**Approach:** Full end-to-end ML project following the standard workflow.

---

## 📋 The ML Project Workflow

| Step | Description |
|------|-------------|
| 1. Frame the problem | Define business objective, select performance measure |
| 2. Get the data | Download, explore, split into train/test |
| 3. Explore & visualize | Understand patterns, correlations, distributions |
| 4. Prepare the data | Clean, transform, engineer features |
| 5. Train models | Try multiple algorithms (Linear, Tree, Forest) |
| 6. Fine-tune | Hyperparameter tuning (GridSearch, RandomizedSearch) |
| 7. Evaluate | Test on unseen data (Test Set) |
| 8. Launch & monitor | Deploy, monitor, retrain |

---

## 📊 Key Steps in Detail

### 1️⃣ Framing the Problem
- **Supervised learning** (labeled data)
- **Regression task** (predict numeric value)
- **Multiple regression** (multiple features)
- **Batch learning** (data fits in memory)
- **Performance measure:** RMSE (Root Mean Square Error)

### 2️⃣ Getting the Data
```python
def load_housing_data():
    # Downloads, extracts, and loads the dataset
    # Returns pandas DataFrame
Key insight: Always create a test set BEFORE you look at the data.

Stratified Sampling
python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
Why? Preserves income category distribution → avoids sampling bias.

3️⃣ Data Exploration (EDA)
Quick Look
python
housing.head()
housing.info()      # 20,640 rows, 10 columns, total_bedrooms has 207 nulls
housing.describe()  # Statistical summary
housing.hist(bins=50, figsize=(12, 8))  # Visualize distributions
Key Observations
Attribute	Issue	Solution
median_income	Capped at 15, scaled (not USD)	Understand the scale
median_house_value	Capped at 500,000	Remove or handle carefully
total_bedrooms	207 missing values	Impute with median
ocean_proximity	Categorical text	One-hot encode
All numeric	Different scales	Feature scaling
Correlations
python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
Feature	Correlation
median_income	0.688 (strongest)
total_rooms	0.137
latitude	-0.140
Insight: median_income is the most important predictor.

4️⃣ Data Preparation (The Pipeline)
Clean Training Set
python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
Numerical Pipeline
python
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
Categorical Pipeline
python
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
Custom Transformers
Ratio Transformer: bedrooms/rooms, rooms/household, people/household

Log Transformer: for heavy-tailed features (population, total_rooms)

Cluster Similarity: RBF similarity to 10 geographic clusters

Full Preprocessing Pipeline
python
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", ClusterSimilarity(n_clusters=10), ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder=num_pipeline)
Output: 24 features ready for training.

5️⃣ Model Training & Evaluation
Linear Regression
python
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
lin_rmse = mean_squared_error(housing_labels, lin_reg.predict(housing), squared=False)
# ~68,687 → Underfitting
Decision Tree
python
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
tree_rmse = cross_val_score(tree_reg, housing, housing_labels,
                            scoring="neg_root_mean_squared_error", cv=10)
# ~66,868 (but 0 on training set → Overfitting!)
Random Forest
python
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmse = cross_val_score(forest_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
# ~47,019 → Much better!
Model Comparison
Model	CV RMSE	Problem
Linear Regression	69,858	Underfitting
Decision Tree	66,868	Overfitting
Random Forest	47,019	Still overfitting but best so far
6️⃣ Fine-Tuning (Hyperparameter Optimization)
Grid Search
python
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
Best Params: n_clusters=15, max_features=6
Best RMSE: 44,042 (improved from 47,019)

Randomized Search
python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'preprocessing__geo__n_clusters': randint(3, 50),
    'random_forest__max_features': randint(2, 20),
}

rnd_search = RandomizedSearchCV(full_pipeline, param_distribs, n_iter=10, cv=3)
rnd_search.fit(housing, housing_labels)
Better for large search spaces – picks random combinations, faster.

7️⃣ Feature Importance Analysis
python
feature_importances = final_model["random_forest"].feature_importances_
sorted(zip(feature_importances, feature_names), reverse=True)
Top Features:

log_median_income (18.7%)

ocean_proximity_INLAND (7.5%)

bedrooms_ratio (6.9%)

8️⃣ Final Evaluation on Test Set
python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)  # 41,424
95% Confidence Interval: [39,275 – 43,467]

9️⃣ Launch, Monitor, Maintain
Save Model
python
import joblib
joblib.dump(final_model, "california_housing_model.pkl")
Load Model
python
final_model_reloaded = joblib.load("california_housing_model.pkl")
predictions = final_model_reloaded.predict(new_data)
MLOps Essentials
Task	Description
Monitoring	Track performance, input data quality
Retraining	Automatically retrain on fresh data
Backups	Save every model & dataset version
Rollback	Ability to revert to previous model
🧠 Key Takeaways
Stratified Sampling prevents sampling bias.

Data preparation is 80% of the work – use pipelines!

Feature engineering (log, ratios, clusters) boosts performance.

Cross-validation gives honest performance estimates.

Hyperparameter tuning (Grid/Randomized Search) improves models.

Ensemble methods (Random Forest) outperform single models.

Test set is for final evaluation ONLY – never tune on it.

Deployment is not the end – monitor and retrain regularly.

💡 Golden Rules
"Garbage in, garbage out" – clean data > complex models.

"Never touch the test set until the end" – avoid data snooping.

"Simple models that generalize beat complex models that memorize."

"The median income is the #1 predictor of housing prices."

🚀 Next Steps
Try SVR, XGBoost, or Neural Networks

Experiment with different feature combinations

Deploy as a REST API or cloud service

Practice on Kaggle competitions