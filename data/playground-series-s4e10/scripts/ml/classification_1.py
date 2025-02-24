import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# 1. Load Data
train_data = pd.read_csv(f"{args.input}/train.csv")
test_data = pd.read_csv(f"{args.input}/test.csv")

# Preprocessing Pipeline
numerical_cols = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
categorical_cols = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

X = train_data.drop(["id", "loan_status"], axis=1)
y = train_data["loan_status"]
X_test = test_data.drop("id", axis=1)

X = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class LoanDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


train_dataset = LoanDataset(X_train, y_train.values)
val_dataset = LoanDataset(X_val, y_val.values)
test_dataset = LoanDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


class LoanClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LoanClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


model = LoanClassifier(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()


train_model(model, train_loader, val_loader)

model.eval()
predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features).squeeze()
        predictions.extend(outputs.numpy())

predictions = [1 if p >= 0.5 else 0 for p in predictions]

submission = pd.DataFrame({"id": test_data["id"], "loan_status": predictions})
submission.to_csv(f"{args.output}/submission.csv", index=False)

print("Submission file 'submission.csv' has been created.")
