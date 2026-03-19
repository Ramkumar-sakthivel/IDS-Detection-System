import pandas as pd
import glob
import numpy as np
import os

# -------------------------------
# 1️⃣ Load all CSV files
# -------------------------------

path = "data/raw/*.csv"
files = glob.glob(path)

if not files:
    print("No CSV files found in data/raw/")
    exit()

df_list = []

for file in files:
    print(f"Loading {file}")
    df = pd.read_csv(file)
    df_list.append(df)

# -------------------------------
# 2️⃣ Merge all files
# -------------------------------

data = pd.concat(df_list, ignore_index=True)
print("\nTotal rows before cleaning:", data.shape)

# -------------------------------
# 3️⃣ Remove duplicates
# -------------------------------

data.drop_duplicates(inplace=True)
print("After removing duplicates:", data.shape)

# -------------------------------
# 4️⃣ Handle Infinite & NaN values
# -------------------------------

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
print("After cleaning NaN & Inf:", data.shape)

# -------------------------------
# 5️⃣ Clean column names (IMPORTANT)
# -------------------------------

data.columns = data.columns.str.strip()

# -------------------------------
# 6️⃣ Select Only Required Features
# -------------------------------

selected_features = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Packet Length Mean",
    "Packet Length Std",
    "Min Packet Length",
    "Max Packet Length",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Init_Win_bytes_forward",
    "act_data_pkt_fwd",
    "Active Mean",
    "Idle Mean",
    "Down/Up Ratio",
    "Label"
]

# Check if any columns are missing
missing = [col for col in selected_features if col not in data.columns]

if missing:
    print("\nERROR: Missing columns detected:")
    print(missing)
    exit()

# Keep only selected columns
data = data[selected_features]

# -------------------------------
# 7️⃣ Convert Label to Binary
# -------------------------------

data['Label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

print("\nFinal dataset shape:", data.shape)
print("\nLabel distribution:")
print(data['Label'].value_counts())

# -------------------------------
# 8️⃣ Save cleaned dataset
# -------------------------------

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/cic_ids2017_selected_features.csv"
data.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved successfully at: {output_path}") 