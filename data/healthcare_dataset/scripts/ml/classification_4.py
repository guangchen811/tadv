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

# Feature engineering: ratio of Billing Amount to Age (if Age != 0), plus existing features
train_df["bill_age_ratio"] = train_df["Billing Amount"] / train_df["Age"].replace(0, np.nan)
test_df["bill_age_ratio"] = test_df["Billing Amount"] / test_df["Age"].replace(0, np.nan)

# Feature Selection
feature_cols = ["Age", "Billing Amount", "Gender", "Medical Condition", "bill_age_ratio"]
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_df[TARGET_COL])

# Identify numerical and categorical columns
num_cols = ["Age", "Billing Amount", "bill_age_ratio"]
cat_cols = ["Gender", "Medical Condition"]

# Scale numeric columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# One-hot encode categorical columns
ohe = OneHotEncoder(handle_unknown="ignore")
X_cat = ohe.fit_transform(X[cat_cols]).toarray()  # Ensure dense format
X_test_cat = ohe.transform(X_test[cat_cols]).toarray()

# Ensure dimensions match before concatenation
print(f"Numeric features shape: {X[num_cols].values.shape}, Categorical features shape: {X_cat.shape}")
print(
    f"Test Numeric features shape: {X_test[num_cols].values.shape}, Test Categorical features shape: {X_test_cat.shape}")

X_combined = np.hstack([X[num_cols].values, X_cat])
X_test_combined = np.hstack([X_test[num_cols].values, X_test_cat])

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
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Model initialization
num_features = X_train_t.shape[1]
num_classes = len(label_encoder.classes_)
model = SimpleMLP(input_dim=num_features, hidden_dim=32, num_classes=num_classes)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
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
        val_acc = (val_preds == y_val_t).float().mean()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

# Inference on test data
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
