class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DataProcessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(drop="first"), categorical_cols)
            ]
        )

    def fit_transform(self, train_data, test_data, target_col, id_col):
        self.target_col = target_col
        self.id_col = id_col
        X = train_data.drop([self.target_col, self.id_col], axis=1)
        y = train_data[self.target_col]
        X_test = test_data.drop(self.id_col, axis=1)

        X = self.preprocessor.fit_transform(X)
        X_test = self.preprocessor.transform(X_test)
        return X, y, X_test

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

class LoanClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LoanClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for features, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss, val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, targets in self.val_loader:
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                y_true.extend(targets.numpy())
                y_pred.extend((outputs.numpy() >= 0.5).astype(int))

        val_acc = accuracy_score(y_true, y_pred)
        return val_loss, val_acc

# Configuration
numerical_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
categorical_cols = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
target_col = "loan_status"
id_col = "id"

# Data Processing
data_processor = DataProcessor(numerical_cols, categorical_cols)
train_data = pd.read_csv("/Kaggle/input/train.csv")
test_data = pd.read_csv("/Kaggle/input/test.csv")

X, y, X_test = data_processor.fit_transform(train_data, test_data, target_col, id_col)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = LoanDataset(X_train, y_train.values)
val_dataset = LoanDataset(X_val, y_val.values)
test_dataset = LoanDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model and Training
model = LoanClassifier(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = Trainer(model, criterion, optimizer, train_loader, val_loader)
trainer.train(epochs=10)

# Prediction
model.eval()
predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features).squeeze()
        predictions.extend((outputs.numpy() >= 0.5).astype(int))

submission = pd.DataFrame({id_col: test_data[id_col], target_col: predictions})
submission.to_csv("/kaggle/output/submission.csv", index=False)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income",
                "cb_person_cred_hist_length", "person_home_ownership", "loan_intent", "cb_person_default_on_file"]

    def used_columns(self):
        pass
