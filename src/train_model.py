import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------

DATA_PATH = "data/processed/cic_ids2017_selected_features.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)


# -------------------------------
# 2️⃣ Split Features & Label
# -------------------------------

X = df.drop("Label", axis=1)
y = df["Label"]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# -------------------------------
# 3️⃣ Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -------------------------------
# 4️⃣ Handle Class Imbalance
# -------------------------------

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("Scale_pos_weight:", scale_pos_weight)


# -------------------------------
# 5️⃣ Train XGBoost Model
# -------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


# -------------------------------
# 6️⃣ Evaluate Model
# -------------------------------

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

fpr = cm[0][1] / (cm[0][0] + cm[0][1])
print("\nFalse Positive Rate (FPR):", fpr)

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)


# -------------------------------
# 7️⃣ Save Model & Feature Names
# -------------------------------

os.makedirs("models", exist_ok=True)

# Save model
pickle.dump(model, open("models/xgboost_model.pkl", "wb"))

# Save feature names (CRITICAL for web app)
pickle.dump(X.columns.tolist(), open("models/feature_names.pkl", "wb"))

print("\nModel saved to models/xgboost_model.pkl")
print("Feature names saved to models/feature_names.pkl")