import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

# Load Data
train_data = pd.read_csv("/Kaggle/input/train.csv")
test_data = pd.read_csv("/Kaggle/input/test.csv")

# Feature Engineering
for df in [train_data, test_data]:
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])  # Log-transform loan amount
    df["cred_hist_ratio"] = df["cb_person_cred_hist_length"] / df["loan_amnt"]  # Credit history to loan ratio

# Select target and predictors
target = "person_income"
numeric_cols = ["log_loan_amnt", "loan_int_rate", "cred_hist_ratio"]
categorical_cols = ["loan_grade", "cb_person_default_on_file", "person_home_ownership"]

X_train = train_data[numeric_cols + categorical_cols]
y_train = train_data[target]
X_test = test_data[numeric_cols + categorical_cols]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# PyTorch Dataset Class
class IncomeDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


train_dataset = IncomeDataset(X_train, y_train)
val_dataset = IncomeDataset(X_val, y_val)
test_dataset = IncomeDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


# Define Regression Model
class IncomeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(IncomeRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = IncomeRegressor(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the Model
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features).squeeze()
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")


train_model(model, train_loader, val_loader)

# Generate Predictions for Test Set
model.eval()
predictions = []
with torch.no_grad():
    for features in test_loader:
        predictions.extend(model(features).squeeze().numpy())

# Save Predictions
submission = pd.DataFrame({"id": test_data.get("id", range(len(test_data))), "predicted_person_income": predictions})
submission.to_csv("/kaggle/output/submission.csv", index=False)
print("Submission saved.")
