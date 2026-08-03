# Key Concepts – Chapter 5 (SVM)

| Term | Definition |
|------|------------|
| **Support Vector** | Data point closest to the decision boundary; determines the margin |
| **Margin** | Distance between decision boundary and support vectors |
| **Large Margin** | Maximizing margin for better generalization |
| **Hard Margin** | No points allowed inside the margin (strict) |
| **Soft Margin** | Allows some margin violations (flexible) |
| **C** | Hyperparameter controlling margin violations; smaller C = simpler model |
| **Kernel Trick** | Computes high-dimensional transformations without adding features |
| **Polynomial Kernel** | Kernel for polynomial features: `(γa·b + r)^d` |
| **RBF Kernel** | Gaussian Radial Basis Function kernel: `exp(-γ||a-b||²)` |
| **γ (gamma)** | Controls influence of each training instance in RBF kernel |
| **Support Vector Machine (SVM)** | Model that finds optimal separating hyperplane |
| **LinearSVC** | Fast linear SVM classifier (no kernel) |
| **SVC** | Full SVM classifier with kernel support |
| **SGDClassifier** | Linear SVM using stochastic gradient descent (supports out-of-core) |
| **Hinge Loss** | Loss function used for SVM: `max(0, 1 - t·s)` |
| **Squared Hinge Loss** | `max(0, 1 - t·s)²` (used by LinearSVC) |
| **Dual Problem** | Alternative formulation enabling kernel trick |
| **Primal Problem** | Original optimization problem |
| **Epsilon (ε)** | Margin width in SVM regression |
| **LinearSVR** | Linear SVM regression |
| **SVR** | SVM regression with kernel support |
| **Out-of-core** | Learning on data that doesn't fit in memory |
| **Decision Function** | `w·x + b` — determines class prediction |