import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# 1. Load Data
train_df = pd.read_csv(f"{args.input}/train.csv")
test_df = pd.read_csv(f"{args.input}/test.csv")

# Rename columns for consistency
train_df = train_df.rename(columns={
    "Age": "age",
    "Gender": "gender",
    "Billing Amount": "billing_amount",
    "Test Results": "test_results"
})
test_df = test_df.rename(columns={
    "Age": "age",
    "Gender": "gender",
    "Billing Amount": "billing_amount"
})

# We drop "Name", "Doctor", "Date of Admission", "Discharge Date" for potential privacy / irrelevance
# We keep "Hospital" just to see if it provides any signal, but normally it could also be dropped
drop_cols = ["Name", "Doctor", "Date of Admission", "Discharge Date", "Room Number"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# We'll create a ratio feature: "cost_per_year_of_age" = billing_amount / age, if age > 0
train_df["cost_per_year_of_age"] = np.where(train_df["age"] == 0, 0, train_df["billing_amount"] / train_df["age"])
test_df["cost_per_year_of_age"] = np.where(test_df["age"] == 0, 0, test_df["billing_amount"] / test_df["age"])

# Setup target and ID
id_col = "id"
target_col = "test_results"
y = train_df[target_col]
lbl = LabelEncoder()
y_enc = lbl.fit_transform(y)

# Drop the target from train
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Distinguish numeric vs. categorical
numeric_cols = []
categorical_cols = []
for c in X.columns:
    if c == id_col:
        continue
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# We'll scale numeric columns and one-hot categorical
scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown="ignore")

X_num = scaler.fit_transform(X[numeric_cols])
X_cat = ohe.fit_transform(X[categorical_cols])
X_combined = np.hstack([X_num, X_cat])

# For test data
test_num = scaler.transform(X_test[numeric_cols])
test_cat = ohe.transform(X_test[categorical_cols])
X_test_combined = np.hstack([test_num, test_cat])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Convert to torch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_combined, dtype=torch.float32)
test_ids = X_test[id_col].values


# Two-layer network
class TwoLayerNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


model = TwoLayerNet(in_dim=X_train_t.shape[1], hid_dim=32, out_dim=len(lbl.classes_))
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# Training
epochs = 6
for ep in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = torch.argmax(val_logits, dim=1)
        val_acc = (val_preds == y_val_t).float().mean()
    print(f"Epoch {ep + 1} | Loss={loss.item():.4f} | Val Acc={val_acc.item():.4f}")

# Final inference
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
