import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# 1. Load Data
train = pd.read_csv(f"{args.input}/train.csv")
test = pd.read_csv(f"{args.input}/test.csv")
train["loan_amnt_cat"] = pd.cut(train["loan_amnt"], bins=[0, 5000, 15000, 30000, float('inf')],
                                labels=["Low", "Med", "High", "VHigh"])
test["loan_amnt_cat"] = pd.cut(test["loan_amnt"], bins=[0, 5000, 15000, 30000, float('inf')],
                               labels=["Low", "Med", "High", "VHigh"])
cols = ["loan_int_rate", "loan_grade", "cb_person_default_on_file", "loan_amnt_cat"]
num, cat = ["loan_int_rate"], ["loan_grade", "cb_person_default_on_file", "loan_amnt_cat"]

# Preprocess
X = train[cols]
y = train["loan_status"]
X_test = test[cols]
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OrdinalEncoder(), cat)])
X, X_test = pre.fit_transform(X), pre.transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class LoanDataset(Dataset):
    def __init__(self, X, y=None): self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y.values,
                                                                                                       dtype=torch.float32) if y is not None else None

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


train_loader = DataLoader(LoanDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(LoanDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(LoanDataset(X_test), batch_size=64)

# Model
model = nn.Sequential(nn.Linear(X_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
criterion, opt = nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    model.train();
    loss_sum = 0
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        loss = criterion(model(X_batch).squeeze(), y_batch)
        loss.backward();
        opt.step();
        loss_sum += loss.item()
    model.eval();
    val_loss = sum(criterion(model(X_batch).squeeze(), y_batch).item() for X_batch, y_batch in val_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {loss_sum:.4f}, Val Loss: {val_loss:.4f}")

# Predict
model.eval()
predictions = [1 if p >= 0.5 else 0 for batch in test_loader for p in model(batch).squeeze().detach().numpy()]
pd.DataFrame({"id": test["id"], "loan_status": predictions}).to_csv(f"{args.output}/submission.csv", index=False)

print("Submission file 'submission.csv' has been created.")
