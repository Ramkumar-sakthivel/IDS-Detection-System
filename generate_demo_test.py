import pandas as pd
import numpy as np

np.random.seed(42)

rows = 500

data = {
    "Flow Bytes/s": np.random.normal(100000, 50000, rows),
    "Flow Packets/s": np.random.normal(20000, 10000, rows),
    "Total Fwd Packets": np.random.randint(1, 50, rows),
    "Total Backward Packets": np.random.randint(1, 30, rows),
    "Total Length of Fwd Packets": np.random.randint(20, 5000, rows),
    "Total Length of Bwd Packets": np.random.randint(20, 4000, rows),
    "Flow IAT Mean": np.random.normal(1e6, 5e5, rows),
    "Flow IAT Std": np.random.normal(1e6, 5e5, rows),
    "Fwd IAT Mean": np.random.normal(1e6, 5e5, rows),
    "Bwd IAT Mean": np.random.normal(2e5, 1e5, rows),
    "SYN Flag Count": np.random.randint(0, 5, rows),
    "RST Flag Count": np.random.randint(0, 2, rows),
    "ACK Flag Count": np.random.randint(0, 5, rows),
    "PSH Flag Count": np.random.randint(0, 5, rows),
    "Packet Length Mean": np.random.normal(300, 150, rows),
    "Packet Length Std": np.random.normal(200, 100, rows),
    "Min Packet Length": np.random.randint(1, 50, rows),
    "Max Packet Length": np.random.randint(50, 1500, rows),
    "Fwd Packet Length Mean": np.random.normal(250, 100, rows),
    "Bwd Packet Length Mean": np.random.normal(200, 80, rows),
    "Init_Win_bytes_forward": np.random.randint(1000, 20000, rows),
    "act_data_pkt_fwd": np.random.randint(0, 5, rows),
    "Active Mean": np.random.normal(100000, 50000, rows),
    "Idle Mean": np.random.normal(1000000, 500000, rows),
    "Down/Up Ratio": np.random.uniform(0, 1, rows)
}

df = pd.DataFrame(data)

# Create some synthetic attack patterns
df["Label"] = np.where(
    (df["Flow Bytes/s"] > 150000) &
    (df["Packet Length Std"] > 250),
    1,
    0
)

df.to_csv("demo_test_flows.csv", index=False)

print("Demo test file generated: demo_test_flows.csv")