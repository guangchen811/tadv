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

target_col = "Test Results"
id_col = "id"
y = train_df[target_col]
train_ids = train_df[id_col].values
test_ids = test_df[id_col].values

# Feature engineering: ratio of Billing Amount to Age (if Age != 0), plus existing features
train_df["bill_age_ratio"] = train_df["Billing Amount"] / train_df["Age"].replace(0, np.nan)
test_df["bill_age_ratio"] = test_df["Billing Amount"] / test_df["Age"].replace(0, np.nan)

# Keep columns: Age, bill_age_ratio, Medical Condition, Admission Type
X = train_df.drop(columns=[target_col, "Name", "Room Number", "Hospital", "Doctor",
                           "Date of Admission", "Discharge Date", id_col, "Insurance Provider", "Gender", "Medication",
                           "Billing Amount"])
X_test = test_df.drop(columns=["Name", "Room Number", "Hospital", "Doctor",
                               "Date of Admission", "Discharge Date", id_col, "Insurance Provider", "Gender",
                               "Medication", "Billing Amount"])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

numeric_cols = ["Age", "bill_age_ratio"]
cat_cols = ["Medical Condition", "Admission Type"]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

ohe = OneHotEncoder(handle_unknown="ignore")
X_cat = ohe.fit_transform(X[cat_cols])
X_test_cat = ohe.transform(X_test[cat_cols])

X_combined = np.hstack([X[numeric_cols].values, X_cat])
X_test_combined = np.hstack([X_test[numeric_cols].values, X_test_cat])

X_train, X_val, y_train, y_val = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42,
                                                  stratify=y_encoded)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_combined, dtype=torch.float32)


class RatioMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(RatioMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_features = X_train_t.shape[1]
num_classes = len(label_encoder.classes_)
model = RatioMLP(num_features, 64, num_classes)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

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
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

test_labels = label_encoder.inverse_transform(test_preds)
submission_df = pd.DataFrame({id_col: test_ids, target_col: test_labels})
submission_df.to_csv(f"{args.output}/submission.csv", index=False)
