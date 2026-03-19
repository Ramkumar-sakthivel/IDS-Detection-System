#  Intrusion Detection System (IDS) using XGBoost

##  Overview

This project implements a **Machine Learning-based Intrusion Detection System (IDS)** using flow-based network features.
It detects malicious network traffic and classifies it as **Normal** or **Attack** using an optimized XGBoost model.

A **SOC-style dashboard** is built using Flask to visualize detection results in a user-friendly interface.

---

##  Features

* Flow-based intrusion detection
* High accuracy classification using XGBoost
* False Positive & False Negative analysis
* Threshold tuning for optimal performance
* Interactive SOC dashboard (Flask web app)
* Upload CSV → Get instant detection results

---

##  Dataset

The model is trained on the **CICIDS2017 dataset**, which contains real-world network traffic including multiple attack types.

### Key Characteristics:

* 2.8+ million records
* 80+ features
* Includes attacks like:

  * DDoS
  * PortScan
  * Web Attacks
  * Infiltration

---

##  Model Details

* Algorithm: **XGBoost Classifier**
* Handling Imbalance: `scale_pos_weight`
* Selected Features: 25 optimized flow-based features

### Final Performance:

* Accuracy: ~99%
* ROC-AUC Score: ~0.9999
* False Positive Rate: <0.001
* Recall: ~0.99

---

##  Project Structure

```
IDS-Detection-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── xgboost_model.pkl
│   └── feature_names.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── Test_File/
│   └── demo_test_flows.csv
```

---

##  Web Dashboard

A **Security Operations Center (SOC)-style dashboard** is implemented using Flask.

### Features:

* Upload CSV file
* Displays:

  * Total records
  * Attack vs Normal traffic
  * Attack percentage
  * Detection status (Secure / Threat Detected)
  * Model evaluation (if labels present)

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the application

```
python app.py
```

### 3️⃣ Open browser

```
http://127.0.0.1:5000
```

### 4️⃣ Upload test file

Use:

```
Test_File/demo_test_flows.csv
```

---

## ⚠️ Important Note

The model works on **flow-based features**, not raw packet data.

### ❌ Not Supported:

* Wireshark CSV (packet-level data)

### ✅ Supported:

* Flow feature CSV (like CICIDS format)

---

##  Analysis Performed

* False Positive Analysis
* False Negative Analysis
* Feature Importance Analysis
* Threshold Optimization

---

##  Demo Data

A synthetic dataset (`demo_test_flows.csv`) is included for demonstration purposes.

---


##  Future Improvements

* Real-time traffic monitoring
* PCAP → Flow feature conversion
* Live dashboard updates
* Deep Learning models

---

##  Keywords

Intrusion Detection System, Machine Learning, XGBoost, Cybersecurity, Network Security, SOC Dashboard
