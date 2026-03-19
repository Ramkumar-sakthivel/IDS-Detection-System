import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

# Load test data
data = pd.read_csv("data/processed/cic_ids2017_selected_features.csv")

X = data.drop("Label", axis=1)
y = data["Label"]

# Load trained model
model = pickle.load(open("models/xgboost_model.pkl", "rb"))

# Use only a subset for faster analysis
X_sample = X.sample(200000, random_state=42)
y_sample = y.loc[X_sample.index]

y_pred = model.predict(X_sample)

cm = confusion_matrix(y_sample, y_pred)
TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix:\n", cm)

false_positives = X_sample[(y_pred == 1) & (y_sample == 0)]

print("\nNumber of False Positives:", len(false_positives))

print("\nMean feature values of False Positives:")
print(false_positives.mean())

print("\nTop 10 important features:")
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df.head(10))