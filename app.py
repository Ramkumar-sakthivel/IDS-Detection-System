from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Load model and features
model = pickle.load(open("models/xgboost_model.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    try:
        df = pd.read_csv(file)

        y_true = None
        if "Label" in df.columns:
            y_true = df["Label"]
            df = df.drop("Label", axis=1)

        df = df[feature_names]
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        predictions = model.predict(df)

        attack_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())
        total = len(predictions)

        attack_percent = round((attack_count / total) * 100, 2)

        status = "Threat Detected" if attack_count > 0 else "System Secure"

        accuracy = None
        cm = None
        if y_true is not None:
            accuracy = round(accuracy_score(y_true, predictions), 4)
            cm = confusion_matrix(y_true, predictions)

        return render_template(
            "result.html",
            total=total,
            attack_count=attack_count,
            normal_count=normal_count,
            attack_percent=attack_percent,
            status=status,
            accuracy=accuracy,
            cm=cm
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)