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

# Drop columns not needed as features
# We're predicting person_income, so remove it from the features
X_train = train_data.drop(["id", "person_income"], axis=1)
y_train = train_data["person_income"]

# For the test set, we'll generate predictions for person_income
X_test = test_data.drop(["id", "person_income"], axis=1, errors='ignore')

# Identify numerical and categorical columns
numerical_cols = [
    "person_age", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
categorical_cols = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file", "loan_status"
]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)


# Create Dataset class
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


train_dataset = IncomeDataset(X_tr, y_tr)
val_dataset = IncomeDataset(X_val, y_val)
test_dataset = IncomeDataset(X_test_processed, None)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


# Define a regression model
class IncomeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(IncomeRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = IncomeRegressor(input_dim=X_tr.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")


train_model(model, train_loader, val_loader)

# Generate predictions on the test set
model.eval()
test_predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features).squeeze().numpy()
        test_predictions.extend(outputs)

# Create a submission DataFrame
# If the test_data doesn't have 'id', we can assume it's provided or re-index
if "id" not in test_data.columns:
    test_data["id"] = np.arange(len(test_data))

submission = pd.DataFrame({
    "id": test_data["id"],
    "predicted_person_income": test_predictions
})

submission.to_csv("/kaggle/output/submission.csv", index=False)
print("Submission saved.")
