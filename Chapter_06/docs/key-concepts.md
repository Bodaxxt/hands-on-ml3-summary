# 🗝️ Key Concepts & Glossary – Chapter 6 (Decision Trees)

A comprehensive dictionary of terms, algorithms, hyperparameters, and mathematical metrics used in Decision Trees.

---

## 📖 Core Terminology (المصطلحات الأساسية)

| Term (المصطلح) | Arabic (المعنى بالعربية) | Definition & Notes (التعريف والملاحظات) |
| :--- | :--- | :--- |
| **Decision Tree** | شجرة القرار | A hierarchical supervised learning model that partitions feature space into distinct rectangular regions using simple decision rules. |
| **Root Node** | العقدة الجذرية | The initial, topmost node in the tree containing the entire training dataset before any splits. |
| **Internal Node** | عقدة داخلية / عقدة قرار | A decision point that evaluates a condition on a specific feature ($x_j \le t_k$) and directs samples to branch left or right. |
| **Leaf Node (Terminal Node)** | عقدة طرفية / ورقة | The final node without children that holds the class distribution or final continuous prediction. |
| **Gini Impurity** | شوائب جيني | A metric of node purity ($0.0 \le G \le 0.5$). Measures probability of misclassifying a randomly chosen element. |
| **Entropy** | الإنتروبيا (مقياس عدم الانتظام) | An information theory metric of disorder ($H = 0$ for pure nodes, $H > 0$ for mixed nodes). |
| **Information Gain** | كسب المعلومات | The reduction in entropy/impurity achieved by splitting a node on a particular feature. |
| **CART** | خوارزمية كارت | *Classification and Regression Trees*. The greedy binary splitting algorithm implemented in Scikit-Learn. |
| **White Box Model** | نموذج الصندوق الأبيض | An interpretable model where internal logic and decision steps are transparent and easily verifiable by humans. |
| **Black Box Model** | نموذج الصندوق الأسود | Complex models (e.g., Deep Neural Networks) whose predictions are hard to interpret or trace directly to simple rules. |
| **Greedy Algorithm** | خوارزمية جشعة | An optimization heuristic that makes the locally optimal choice at each step without considering future global outcomes. |
| **NP-Complete** | معضلة صعبة حسابياً | Computational complexity class indicating that finding the optimal tree is mathematically intractable ($\mathcal{O}(\exp(m))$). |
| **Non-parametric Model** | نموذج غير معلمي | A model that does not assume a fixed mathematical form/distribution for the data prior to training. |
| **Parametric Model** | نموذج معلمي | A model with a fixed number of parameters (like Linear/Logistic Regression: $y = \theta^T X$). |
| **Orthogonal Boundaries** | حدود متعامدة | Decision boundaries aligned strictly with coordinate axes ($x_1 \le c$), making trees sensitive to data rotation. |
| **PCA (Principal Component Analysis)** | تحليل المكونات الرئيسية | Dimensionality reduction technique often applied before Decision Trees to rotate features toward maximum variance. |

---

## ⚙️ Scikit-Learn Regularization Hyperparameters

| Hyperparameter | Type / Default | Effect of Tuning (تأثير التعديل) |
| :--- | :--- | :--- |
| `max_depth` | `int`, `None` | Restricts the maximum tree depth. **Decreasing** this reduces overfitting. |
| `min_samples_split` | `int` or `float`, `2` | Minimum samples a node must contain before it can be split. **Increasing** regularizes the model. |
| `min_samples_leaf` | `int` or `float`, `1` | Minimum samples required in any leaf node. **Increasing** prevents tiny isolated leaf partitions. |
| `max_leaf_nodes` | `int`, `None` | Caps total number of leaf nodes created. |
| `max_features` | `int`, `float`, `None` | Limits maximum number of features evaluated for best split at each node. |
| `criterion` | `'gini'`, `'entropy'`, `'log_loss'` | Impurity measure for classification (`'squared_error'`, `'absolute_error'` for regression). |

---

## 🧮 Mathematical Formulas

### 1. Gini Impurity:
$$G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2$$

### 2. Entropy:
$$H_i = -\sum_{k=1, p_{i,k} \ne 0}^{n} p_{i,k} \log_2(p_{i,k})$$

### 3. Classification Cost Function (CART):
$$J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$

### 4. Regression Mean Squared Error (Node MSE):
$$\text{MSE}_{\text{node}} = \frac{1}{m_{\text{node}}} \sum_{i \in \text{node}} (\hat{y}_{\text{node}} - y^{(i)})^2$$
