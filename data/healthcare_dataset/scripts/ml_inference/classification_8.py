import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# 1. Load Data
train_df = pd.read_csv(f"{args.input}/train.csv")
test_df = pd.read_csv(f"{args.input}/test.csv")

# Rename columns
train_df = train_df.rename(columns={
    "Date of Admission": "admission_dt",
    "Discharge Date": "discharge_dt",
    "Test Results": "test_results"
})
test_df = test_df.rename(columns={
    "Date of Admission": "admission_dt",
    "Discharge Date": "discharge_dt"
})

# Explanation: We'll drop "Name" and "Gender" (potential privacy), "Doctor" (too specific).
drop_cols = ["Name", "Gender", "Doctor", "Room Number", "Insurance Provider"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

train_df = train_df.dropna()

# Create a length_of_stay feature in days
train_df["admission_dt"] = pd.to_datetime(train_df["admission_dt"], errors="coerce")
train_df["discharge_dt"] = pd.to_datetime(train_df["discharge_dt"], errors="coerce")
train_df["length_of_stay"] = (train_df["discharge_dt"] - train_df["admission_dt"]).dt.days

test_df["admission_dt"] = pd.to_datetime(test_df["admission_dt"], errors="coerce")
test_df["discharge_dt"] = pd.to_datetime(test_df["discharge_dt"], errors="coerce")
test_df["length_of_stay"] = (test_df["discharge_dt"] - test_df["admission_dt"]).dt.days

# We'll remove the original date columns, as we only keep length_of_stay
train_df = train_df.drop(columns=["admission_dt", "discharge_dt"], errors="ignore")
test_df = test_df.drop(columns=["admission_dt", "discharge_dt"], errors="ignore")

#
train_df = train_df.drop(columns=["Blood Type"], errors="ignore")
test_df = test_df.drop(columns=["Blood Type"], errors="ignore")
train_df = train_df.dropna()

id_col = "id"
target_col = "test_results"
y = train_df[target_col]
lbl = LabelEncoder()
y_enc = lbl.fit_transform(y)

X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Identify numeric vs. categorical
numeric_cols = []
categorical_cols = []
for c in X.columns:
    if c == id_col:
        continue
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# Fill missing numeric with 0 or mean
X[numeric_cols] = X[numeric_cols].fillna(0)
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

scaler = StandardScaler()
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_num = scaler.fit_transform(X[numeric_cols])
X_cat = oe.fit_transform(X[categorical_cols]).reshape(-1, len(categorical_cols))
X_combined = np.hstack([X_num, X_cat])

test_num = scaler.transform(X_test[numeric_cols])
test_cat = oe.transform(X_test[categorical_cols]).reshape(-1, len(categorical_cols))
X_test_combined = np.hstack([test_num, test_cat])

X_train, X_val, y_train, y_val = train_test_split(X_combined, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_combined, dtype=torch.float32)
test_ids = X_test[id_col].values


# Define a deeper MLP with dropout
class DeeperMLP(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, out_dim):
        super(DeeperMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


model = DeeperMLP(in_dim=X_train_t.shape[1], h1_dim=64, h2_dim=32, out_dim=len(lbl.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Train
epochs = 7
for ep in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_preds = torch.argmax(val_out, dim=1)
        val_acc = (val_preds == y_val_t).float().mean()
    print(f"Epoch {ep + 1} | Loss={loss.item():.4f} | Val Acc={val_acc.item():.4f}")

# Prediction
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

test_labels = lbl.inverse_transform(test_preds)
submission_df = pd.DataFrame({
    id_col: test_ids,
    "Test Results": test_labels
})
submission_df.to_csv(f"{args.output}/submission.csv", index=False)
