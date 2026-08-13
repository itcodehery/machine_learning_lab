# ML For Social Good: Breast Cancer Detection

## 1. Mission Overview
**Problem:** Breast cancer is one of the most common cancers among women worldwide. Early detection is critical for improving survival rates and reducing the severity of required treatments.
**Solution:** An automated, highly accurate diagnostic tool using advanced ensemble machine learning to classify breast mass features (from Fine Needle Aspirates) as benign or malignant.
**Value:** Reduces diagnostic errors, provides a reliable second opinion for pathologists, and ensures rapid processing of biopsy results, directly contributing to better patient outcomes.

## 2. Dataset
We will use the **Breast Cancer Wisconsin (Diagnostic) Data Set** from the UCI Machine Learning Repository.
- **Source:** `https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`
- **Features:** 30 real-valued features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Target:** Diagnosis (M = malignant, B = benign).

## 3. Reproducibility & Data Fetching
To ensure the solution is reproducible, the dataset can be fetched directly via `curl`. 

```bash
# Create project directory
mkdir breast_cancer_ml && cd breast_cancer_ml
mkdir data

# Fetch the dataset directly from UCI
curl -o data/wdbc.data https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```

## 4. Advanced Ensemble Machine Learning Strategy
We will engineer a robust predictive model using multiple ensemble techniques to maximize recall (minimizing false negatives, which is crucial in medical diagnosis):

1. **Bagging:** Random Forest Classifier to handle feature interactions and reduce variance.
2. **Boosting:** Gradient Boosting (or XGBoost) to iteratively correct errors and minimize bias.
3. **Stacking/Voting:** A Soft Voting Classifier or Stacking Classifier that combines the predictions of Random Forest, Gradient Boosting, and a robust linear model (like Logistic Regression) to achieve superior generalization.

## 5. Draft Implementation (Python)
Save the following script as `main.py` and run it to train and evaluate the models.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import urllib.request
import os

# 1. Data Loading Pipeline
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
DATA_PATH = "data/wdbc.data"

if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

# Define column names based on dataset description
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = pd.read_csv(DATA_PATH, header=None, names=columns)

# 2. Data Preprocessing
# Drop ID column as it's not predictive
X = df.drop(['id', 'diagnosis'], axis=1)
# Encode diagnosis: M = 1 (Malignant), B = 0 (Benign)
y = df['diagnosis'].map({'M': 1, 'B': 0})

# Split the data (Stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features (critical for linear models and distance metrics in ensembles)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Engineering: Ensemble Setup
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr_clf = LogisticRegression(max_iter=1000, random_state=42)

# Create a Voting Classifier (Soft voting uses predicted probabilities)
ensemble_clf = VotingClassifier(
    estimators=[
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('lr', lr_clf)
    ],
    voting='soft'
)

# 4. Training and Evaluation
print("Training Ensemble Model...")
ensemble_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = ensemble_clf.predict(X_test_scaled)

# 5. Results & Defensibility
print("\\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\nClassification Report:")
# In medical contexts, we care deeply about Recall for the positive class (Malignant)
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## 6. Next Steps for Defensibility
- **Cross-Validation:** Implement k-fold cross-validation to prove the model's stability across different data splits.
- **Hyperparameter Tuning:** Use Grid Search or Optuna to fine-tune the ensemble components.
- **Explainability:** Integrate SHAP (SHapley Additive exPlanations) to explain which features (e.g., cell radius, texture) drive the model's decisions, building trust with medical professionals.
