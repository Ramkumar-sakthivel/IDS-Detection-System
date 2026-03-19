import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("data/processed/cic_ids2017_selected_features.csv")

X = data.drop("Label", axis=1)
y = data["Label"]

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

threshold = 0.8
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > threshold).astype(int)

# -------------------------------
# Identify False Negatives
# -------------------------------

X_test_copy = X_test.copy()
X_test_copy["Actual"] = y_test.values
X_test_copy["Predicted"] = y_pred

false_negatives = X_test_copy[
    (X_test_copy["Actual"] == 1) &
    (X_test_copy["Predicted"] == 0)
]

true_positives = X_test_copy[
    (X_test_copy["Actual"] == 1) &
    (X_test_copy["Predicted"] == 1)
]

print("Number of False Negatives:", len(false_negatives))

# -------------------------------
# Compare Feature Means
# -------------------------------

fn_means = false_negatives.drop(["Actual", "Predicted"], axis=1).mean()
tp_means = true_positives.drop(["Actual", "Predicted"], axis=1).mean()

comparison = pd.DataFrame({
    "False_Negative_Mean": fn_means,
    "True_Positive_Mean": tp_means
})

print("\nTop 10 Features with Largest Difference:\n")

comparison["Difference"] = abs(
    comparison["False_Negative_Mean"] - comparison["True_Positive_Mean"]
)

print(
    comparison.sort_values(by="Difference", ascending=False).head(10)
)