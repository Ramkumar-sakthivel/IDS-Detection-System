import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

# Load data
data = pd.read_csv("data/processed/cic_ids2017_selected_features.csv")

X = data.drop("Label", axis=1)
y = data["Label"]

# Sample for faster processing
X_sample = X.sample(200000, random_state=42)
y_sample = y.loc[X_sample.index]

# Load model
model = pickle.load(open("models/xgboost_model.pkl", "rb"))

# Get probabilities
y_probs = model.predict_proba(X_sample)[:, 1]

thresholds = [0.72, 0.75, 0.78, 0.8]

for t in thresholds:
    y_pred = (y_probs > t).astype(int)
    cm = confusion_matrix(y_sample, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    fpr = FP / (FP + TN)
    recall = TP / (TP + FN)
    
    print(f"\nThreshold: {t}")
    print("FPR:", round(fpr, 6))
    print("Recall:", round(recall, 6))