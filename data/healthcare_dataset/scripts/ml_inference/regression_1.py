import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# 1. Load Data
train_df = pd.read_csv(f"{args.input}/train.csv")
test_df = pd.read_csv(f"{args.input}/test.csv")

# Rename target column
train_df = train_df.rename(columns={"Billing Amount": "billing_amount"})
test_df = test_df.rename(columns={"Billing Amount": "billing_amount"})

target_col = "billing_amount"
id_col = "id"

# Drop unnecessary columns
drop_cols = ["Name", "Test Results", "Doctor", "Discharge Date", "Date of Admission"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# Feature engineering: Create cost_age_ratio feature
train_df["cost_age_ratio"] = train_df["billing_amount"] / train_df["Age"].replace(0, np.nan)
train_df["cost_age_ratio"] = train_df["cost_age_ratio"].fillna(0)

# Extract target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])

# Remove ID column if present
if id_col in X.columns:
    X = X.drop(columns=[id_col])
test_ids = test_df[id_col].values
test_df = test_df.drop(columns=[id_col], errors="ignore") if id_col in test_df.columns else test_df

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Scale numeric columns
scaler = StandardScaler()
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_num = scaler.fit_transform(X[numeric_cols])
X_cat = oe.fit_transform(X[categorical_cols])

# Ensure both arrays have the same number of dimensions before stacking
X_combined = np.hstack([X_num, X_cat])

# Process test data
test_numeric = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0) if set(numeric_cols).issubset(
    test_df.columns) else np.zeros((len(test_df), len(numeric_cols)))
test_categorical = test_df[categorical_cols].astype(str) if set(categorical_cols).issubset(
    test_df.columns) else np.zeros((len(test_df), len(oe.get_feature_names_out())))

testX_num = scaler.transform(test_numeric) if not isinstance(test_numeric, np.ndarray) else test_numeric
testX_cat = oe.transform(test_categorical) if not isinstance(test_categorical,
                                                             np.ndarray) else test_categorical

testX_combined = np.hstack([testX_num, testX_cat])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(testX_combined, dtype=torch.float32)


# Define model
class TwoLayerRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train model
model = TwoLayerRegressor(input_dim=X_train.shape[1], hidden_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

for epoch in range(20):
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
    print(f"Epoch {epoch + 1}, Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

# Predict
test_preds = np.array(model(X_test_t).view(-1).detach().tolist())
submission = pd.DataFrame({"id": test_ids, "Billing Amount": test_preds})
submission.to_csv(f"{args.output}/submission.csv", index=False)
