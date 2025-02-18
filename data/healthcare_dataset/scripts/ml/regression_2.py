import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Rename "Billing Amount" -> "billing_amount" for clarity
train_df = train_df.rename(columns={"Billing Amount": "billing_amount",
                                    "Date of Admission": "admission_dt",
                                    "Discharge Date": "discharge_dt"})
test_df = test_df.rename(columns={"Billing Amount": "billing_amount",
                                  "Date of Admission": "admission_dt",
                                  "Discharge Date": "discharge_dt"})

target_col = "billing_amount"
id_col = "id"

# Create length_of_stay
train_df["admission_dt"] = pd.to_datetime(train_df["admission_dt"], errors="coerce")
train_df["discharge_dt"] = pd.to_datetime(train_df["discharge_dt"], errors="coerce")
train_df["length_of_stay"] = (train_df["discharge_dt"] - train_df["admission_dt"]).dt.days

test_df["admission_dt"] = pd.to_datetime(test_df["admission_dt"], errors="coerce")
test_df["discharge_dt"] = pd.to_datetime(test_df["discharge_dt"], errors="coerce")
test_df["length_of_stay"] = (test_df["discharge_dt"] - test_df["admission_dt"]).dt.days

# Drop original date columns
train_df = train_df.drop(columns=["admission_dt", "discharge_dt", "Test Results", "Doctor", "Name"], errors="ignore")
test_df = test_df.drop(columns=["admission_dt", "discharge_dt", "Test Results", "Doctor", "Name"], errors="ignore")

# Fill missing numeric
train_df["length_of_stay"] = train_df["length_of_stay"].fillna(0)
test_df["length_of_stay"] = test_df["length_of_stay"].fillna(0)

# Target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])

# ID
test_ids = test_df[id_col].values if id_col in test_df.columns else None
if id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col in test_df.columns:
    test_df = test_df.drop(columns=[id_col])

# Numeric columns
numeric_cols = []
categorical_cols = []
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# Convert categorical columns to numeric codes if they exist
for cat in categorical_cols:
    train_df[cat] = train_df[cat].astype(str)  # just in case
    test_df[cat] = test_df[cat].astype(str)  # just in case
    unique_vals = list(set(train_df[cat].unique()).union(set(test_df[cat].unique())))
    enc_map = {val: i for i, val in enumerate(unique_vals)}
    X[cat] = train_df[cat].map(enc_map)
    test_df[cat] = test_df[cat].map(enc_map)

# Scale numeric
scaler = StandardScaler()
X_num = scaler.fit_transform(X[numeric_cols])
test_num = scaler.transform(test_df[numeric_cols])

# Combine numeric + categorical
X_cat = X[categorical_cols].values if categorical_cols else np.zeros((len(X), 0))
test_cat = test_df[categorical_cols].values if categorical_cols else np.zeros((len(test_df), 0))

X_combined = np.hstack([X_num, X_cat])
testX_combined = np.hstack([test_num, test_cat])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(testX_combined, dtype=torch.float32)


# Simple multi-layer MLP with dropout
class StayMLP(nn.Module):
    def __init__(self, in_dim, h_dim=64):
        super(StayMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_dim, 1)  # 1 for regression
        )

    def forward(self, x):
        return self.net(x)


model = StayMLP(in_dim=X_train_t.shape[1], h_dim=64)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t)
    print(f"Epoch {epoch + 1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).view(-1).numpy()

submission = pd.DataFrame({
    "id": test_ids,
    "Billing Amount": test_preds
})
submission.to_csv("submission.csv", index=False)
