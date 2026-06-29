# 🛠️ Solutions – Chapter 3 Exercises

This document contains full solution code and brief explanations for the exercises in Chapter 3 of the machine learning workbook.

---

## 🧩 Exercise 1: MNIST Classifier >97% Accuracy

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Define model
knn_clf = KNeighborsClassifier()

# Hyperparameter grid
param_grid = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': [3, 4, 5, 6, 7]
}

# Grid search
grid_search = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Results
print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# Test evaluation
best_knn = grid_search.best_estimator_
test_acc = best_knn.score(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")  # > 0.97 ✅
```

> [!NOTE]
> **Expected Outcome:** Accuracy > 97% with `weights='distance'` and `n_neighbors=4`.

---

## 🧩 Exercise 2: Data Augmentation (Shifted Images)

```python
from scipy.ndimage import shift
import numpy as np

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted = shift(image, [dy, dx], cval=0, mode='constant')
    return shifted.reshape([-1])

# Expand training set
X_train_expanded = []
y_train_expanded = []

for X, y in zip(X_train, y_train):
    X_train_expanded.append(X)
    y_train_expanded.append(y)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        X_train_expanded.append(shift_image(X, dx, dy))
        y_train_expanded.append(y)

X_train_expanded = np.array(X_train_expanded)
y_train_expanded = np.array(y_train_expanded)

# Train on expanded data
knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_clf.fit(X_train_expanded, y_train_expanded)

# Evaluate
test_acc = knn_clf.score(X_test, y_test)
print(f"Test accuracy with augmentation: {test_acc:.4f}")  # > 97% ✅
```

> [!NOTE]
> **Result:** Accuracy improves significantly (often > 97.5%).

---

## 🧩 Exercise 3: Titanic Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = pd.get_dummies(train_data[features], drop_first=True)
y = train_data['Survived']

# Handle missing values
X = X.fillna(X.median())

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_val_scaled)
print(f"Validation accuracy: {accuracy_score(y_val, y_pred):.4f}")
```

> [!NOTE]
> **Expected Outcome:** ~80-83% validation accuracy.

---

## 🧩 Exercise 4: Spam Classifier

```python
import email
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load emails (assumes files are in spam/ and ham/ directories)
def load_emails_from_folder(folder, label):
    import os
    emails = []
    labels = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as f:
            emails.append(f.read())
            labels.append(label)
    return emails, labels

# Load data
spam_emails, spam_labels = load_emails_from_folder('spam', 1)
ham_emails, ham_labels = load_emails_from_folder('ham', 0)

emails = spam_emails + ham_emails
labels = spam_labels + ham_labels

# Vectorize
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(emails)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

> [!NOTE]
> **Expected Outcome:** High precision and recall (> 0.95).

---

## 📊 Summary of Solutions

| Exercise | Topic | Key Skill |
| :---: | :--- | :--- |
| **1** | MNIST | Hyperparameter tuning with GridSearch |
| **2** | Data Augmentation | Image shifting to improve accuracy |
| **3** | Titanic | End-to-end classification on real data |
| **4** | Spam Classifier | Text preprocessing and Naive Bayes |

---

> [!TIP]
> *"Practice is the only way to master machine learning. These exercises cover the core skills you'll need in real projects."*