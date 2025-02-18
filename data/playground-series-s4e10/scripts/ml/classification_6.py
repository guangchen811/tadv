import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

# Load Data
train_data = pd.read_csv("/Kaggle/input/train.csv")
test_data = pd.read_csv("/Kaggle/input/test.csv")

# Select columns for modeling
selected_cols = [
    "person_age", "person_income", "loan_amnt", "loan_int_rate", "loan_grade",
    "person_home_ownership", "cb_person_default_on_file"
]
target_col = "loan_status"
id_col = "id"

# Feature Engineering: Adding Loan-to-Income Ratio
train_data["loan_to_income_ratio"] = train_data["loan_amnt"] / train_data["person_income"]
test_data["loan_to_income_ratio"] = test_data["loan_amnt"] / test_data["person_income"]

# Update selected columns
selected_cols.append("loan_to_income_ratio")

# Preprocessing Pipeline
numerical_cols = ["person_age", "person_income", "loan_amnt", "loan_int_rate", "loan_to_income_ratio"]
categorical_cols = ["loan_grade", "person_home_ownership", "cb_person_default_on_file"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

X = train_data[selected_cols]
y = train_data[target_col]
X_test = test_data[selected_cols]

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


# Training Loop
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
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                y_true.extend(targets.numpy())
                y_pred.extend((outputs.numpy() >= 0.5).astype(int))

        val_accuracy = accuracy_score(y_true, y_pred)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


train_model(model, train_loader, val_loader)

# Prediction
model.eval()
predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features).squeeze()
        predictions.extend((outputs.numpy() >= 0.5).astype(int))

submission = pd.DataFrame({id_col: test_data[id_col], target_col: predictions})
submission.to_csv("/kaggle/output/submission.csv", index=False)
