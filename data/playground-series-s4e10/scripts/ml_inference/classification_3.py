import argparse

import numpy as np
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

# Rename columns for clarity and consistency
train_data = train_data.rename(columns={
    "person_age": "age",
    "person_income": "income",
    "person_home_ownership": "home_ownership",
    "person_emp_length": "emp_length",
    "loan_amnt": "loan_amount",
    "loan_int_rate": "interest_rate",
    "loan_percent_income": "loan_income_ratio",
    "cb_person_cred_hist_length": "credit_hist_length",
    "loan_grade": "grade",
    "loan_intent": "intent",
    "cb_person_default_on_file": "default_flag"
})
test_data = test_data.rename(columns={
    "person_age": "age",
    "person_income": "income",
    "person_home_ownership": "home_ownership",
    "person_emp_length": "emp_length",
    "loan_amnt": "loan_amount",
    "loan_int_rate": "interest_rate",
    "loan_percent_income": "loan_income_ratio",
    "cb_person_cred_hist_length": "credit_hist_length",
    "loan_grade": "grade",
    "loan_intent": "intent",
    "cb_person_default_on_file": "default_flag"
})

# Create a derived feature that's often useful: income-to-loan ratio
train_data["income_to_loan"] = train_data["income"] / (train_data["loan_amount"] + 1e-9)
test_data["income_to_loan"] = test_data["income"] / (test_data["loan_amount"] + 1e-9)

# Drop some columns that are generally not used as features
# For example, 'id' is not a feature, and 'intent' might be less predictive in this scenario
train_data = train_data.drop(columns=["id", "intent"], errors='ignore')
test_data = test_data.drop(columns=["id", "intent"], errors='ignore')

# Let's say we want to drop columns with very low variance (not helpful)
# This is a common step before modeling
numeric_cols_for_variance = ["age", "income", "emp_length", "loan_amount", "interest_rate",
                             "loan_income_ratio", "credit_hist_length", "income_to_loan"]
variance_threshold = 1e-4
low_var_cols = []
for col in numeric_cols_for_variance:
    if train_data[col].var() < variance_threshold:
        low_var_cols.append(col)

# Drop low variance columns from both train and test
train_data = train_data.drop(columns=low_var_cols, errors='ignore')
test_data = test_data.drop(columns=low_var_cols, errors='ignore')

categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train_data[col], _ = train_data[col].factorize()
    test_data[col] = test_data[col].map({v: i for i, v in enumerate(train_data[col].unique())}).fillna(-1)

# Compute correlation matrix and handle NaN values
corr_matrix = train_data.corr().abs()

# Fill NaNs with 0 before idxmax() to prevent KeyError
corr_matrix = corr_matrix.fillna(0)

# Select upper triangle of correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Drop all-NaN columns (those with no correlation info)
upper_triangle = upper_triangle.dropna(axis=1, how="all")

# Identify highly correlated pairs (correlation > 0.95)
high_corr_pairs = [(col1, col2) for col1 in upper_triangle.columns
                   for col2 in upper_triangle.columns
                   if col1 != col2 and upper_triangle.loc[col1, col2] > 0.95]

# Drop the second column in each highly correlated pair
for col1, col2 in high_corr_pairs:
    if col2 in train_data.columns:
        train_data.drop(columns=[col2], errors='ignore', inplace=True)
        test_data.drop(columns=[col2], errors='ignore', inplace=True)

# At this point, we have a set of numeric and categorical columns left.
# Let's identify them:
all_cols = train_data.columns.tolist()
all_cols.remove("loan_status")  # target

# Assume anything not numeric is categorical
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(train_data[c])]
categorical_cols = [c for c in all_cols if c not in numeric_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

X = train_data.drop("loan_status", axis=1)
y = train_data["loan_status"]
X_test = test_data.copy()

X = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test)

# Train-validation split
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


# Simple neural network
class LoanClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LoanClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
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

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        # In a realistic scenario, we might print or log the losses here


train_model(model, train_loader, val_loader)

model.eval()
predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features).squeeze()
        predictions.extend(outputs.numpy())

predictions = [1 if p >= 0.5 else 0 for p in predictions]

# If 'id' was dropped, we must ensure test_data still has it. If not, let's assume we reset the index as id
# In a real scenario, ensure 'id' is preserved or handle it appropriately.
if "id" not in test_data.columns:
    # This is a fallback scenario. In a real-world pipeline, you'd keep 'id' separate.
    test_data["id"] = np.arange(len(test_data))

submission = pd.DataFrame({"id": test_data["id"], "loan_status": predictions})
submission.to_csv(f"{args.output}/submission.csv", index=False)
