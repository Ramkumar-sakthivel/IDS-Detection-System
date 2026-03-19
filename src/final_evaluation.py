import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("data/processed/cic_ids2017_selected_features.csv")

X = data.drop("Label", axis=1)
y = data["Label"]

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------
# Load Model
# -------------------------------

model = pickle.load(open("models/xgboost_model.pkl", "rb"))

# -------------------------------
# Apply Threshold = 0.8
# -------------------------------

threshold = 0.75

y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > threshold).astype(int)

# -------------------------------
# Evaluation
# -------------------------------

print(f"\nUsing Threshold: {threshold}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix:")
print(cm)

fpr = FP / (FP + TN)
recall = TP / (TP + FN)
roc_auc = roc_auc_score(y_test, y_probs)

print("\nFalse Positive Rate (FPR):", fpr)
print("Recall:", recall)
print("ROC-AUC Score:", roc_auc)