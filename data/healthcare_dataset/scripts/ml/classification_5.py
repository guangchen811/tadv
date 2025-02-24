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

# Define key columns
ID_COL = "id"
TARGET_COL = "Test Results"

# Ensure ID column exists
if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
    raise ValueError("ID column is missing from the dataset. Check the input files.")

# Retain IDs separately
train_ids = train_df[ID_COL].values
test_ids = test_df[ID_COL].values

# Convert date columns to pandas datetime
train_df["Date of Admission"] = pd.to_datetime(train_df["Date of Admission"], errors="coerce")
train_df["Discharge Date"] = pd.to_datetime(train_df["Discharge Date"], errors="coerce")
test_df["Date of Admission"] = pd.to_datetime(test_df["Date of Admission"], errors="coerce")
test_df["Discharge Date"] = pd.to_datetime(test_df["Discharge Date"], errors="coerce")

# Feature engineering: length_of_stay in days
train_df["length_of_stay"] = (train_df["Discharge Date"] - train_df["Date of Admission"]).dt.days.fillna(0)
test_df["length_of_stay"] = (test_df["Discharge Date"] - test_df["Date of Admission"]).dt.days.fillna(0)

# Feature Selection
feature_cols = ["Billing Amount", "Age", "length_of_stay", "Blood Type", "Medical Condition"]
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_df[TARGET_COL])

# Identify numerical and categorical columns
numeric_cols = ["Billing Amount", "Age", "length_of_stay"]
cat_cols = ["Blood Type", "Medical Condition"]

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols].fillna(0))
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols].fillna(0))

# One-hot encode categorical columns
ohe = OneHotEncoder(handle_unknown="ignore")
X_cat = ohe.fit_transform(X[cat_cols]).toarray()
X_test_cat = ohe.transform(X_test[cat_cols]).toarray()

# Ensure dimensions match before concatenation
X_combined = np.hstack([X[numeric_cols].values, X_cat])
X_test_combined = np.hstack([X_test[numeric_cols].values, X_test_cat])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42,
                                                  stratify=y_encoded)

# Convert data to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_combined, dtype=torch.float32)


# Define a simple MLP model
class StayMLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(StayMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# Model initialization
num_features = X_train_t.shape[1]
num_classes = len(label_encoder.classes_)
model = StayMLP(num_features, 32, num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
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
        accuracy = (val_preds == y_val_t).float().mean()
    print(f"Epoch {epoch + 1}, Loss={loss.item():.4f}, Val Accuracy={accuracy.item():.4f}")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = np.array(torch.argmax(test_logits, dim=1).tolist())

# Decode predictions
test_labels = label_encoder.inverse_transform(test_preds)

# Save predictions
submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_labels})
submission_df.to_csv(f"{args.output}/submission.csv", index=False)
print("Created submission.csv!")
