class KaggleLoanColumnDetectionTask:
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

# Load Data
train_data = pd.read_csv("/Kaggle/input/train.csv")
test_data = pd.read_csv("/Kaggle/input/test.csv")

# Original columns (for reference, not shown in final code):
# [
#   "id", "person_age", "person_income", "person_home_ownership",
#   "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
#   "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
#   "cb_person_cred_hist_length", "loan_status"
# ]

# Shuffle columns to break any recognizable pattern
train_data = train_data.sample(frac=1, axis=1, random_state=42)

# Drop the identified id columns if found
train_data = train_data.drop(columns=['id'], errors='ignore')
test_data = test_data.drop(columns=['id'], errors='ignore')

# Now, potential_target should be person_income
target = potential_target
y_train = train_data[target]
X_train = train_data.drop(target, axis=1)
X_test = test_data.drop(target, axis=1, errors='ignore')

# Identify numeric and categorical columns by dtype
numeric_candidates = []
categorical_candidates = []
for c in X_train.columns:
    if pd.api.types.is_numeric_dtype(X_train[c]):
        numeric_candidates.append(c)
    else:
        # Assume non-numeric are categorical
        categorical_candidates.append(c)

# Drop numeric columns with almost no variance
low_var_threshold = 1e-4
low_var_cols = [c for c in numeric_candidates if X_train[c].var() < low_var_threshold]
X_train = X_train.drop(columns=low_var_cols, errors='ignore')
X_test = X_test.drop(columns=low_var_cols, errors='ignore')
numeric_candidates = [c for c in numeric_candidates if c not in low_var_cols]

# Check correlation among remaining numeric features, drop highly correlated ones
if len(numeric_candidates) > 1:
    corr_matrix = X_train[numeric_candidates].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Let's define a correlation threshold
    corr_threshold = 0.95
    highly_correlated = [col2 for col1 in upper_triangle.columns for col2 in upper_triangle.columns if col1 != col2 and upper_triangle.loc[col1, col2] > corr_threshold]
    highly_correlated = list(set(highly_correlated))  # unique
    X_train = X_train.drop(columns=highly_correlated, errors='ignore')
    X_test = X_test.drop(columns=highly_correlated, errors='ignore')
    numeric_candidates = [c for c in numeric_candidates if c not in highly_correlated]

# Let's say we also have a logic to drop categorical columns that appear less than a certain frequency.
# This might represent rare categories. We'll drop entire columns that violate some condition (artificial example).
for cat_col in categorical_candidates:
    # If median frequency of top categories is too low, drop column
    if cat_col in X_train.columns and pd.api.types.is_categorical_dtype(X_train[cat_col]):
        # Convert to category if not categorical
        pass
    # Let's assume if there's a dominant category >90%, we still keep it,
    # else if it's too uniform, we drop. Just a random condition:
    if cat_col in X_train.columns:
        top_cat_freq = X_train[cat_col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_cat_freq < 0.5:
            # Drop this categorical column to simulate complexity
            X_train = X_train.drop(columns=[cat_col], errors='ignore')
            X_test = X_test.drop(columns=[cat_col], errors='ignore')

categorical_candidates = [c for c in categorical_candidates if c in X_train.columns]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [c for c in numeric_candidates if c in X_train.columns]),
        ("cat", OneHotEncoder(drop="first"), [c for c in categorical_candidates if c in X_train.columns])
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

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

def train_model(model, train_loader, val_loader, epochs=5):
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

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")

train_model(model, train_loader, val_loader)

model.eval()
test_predictions = []
with torch.no_grad():
    for features in test_loader:
        outputs = model(features).squeeze().numpy()
        test_predictions.extend(outputs)

# If 'id' is not known, create a pseudo-id for submission
if "col_0" not in test_data.columns:
    test_data["col_0"] = np.arange(len(test_data))

submission = pd.DataFrame({
    "id": test_data["col_0"],  # assuming col_0 was id-like
    "predicted_person_income": test_predictions
})
submission.to_csv("/kaggle/output/submission.csv", index=False)
print("Submission saved.")
"""

    def target_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_amnt', 'loan_grade', 'loan_int_rate',
                'loan_intent', 'loan_percent_income', 'loan_status', 'person_age', 'person_emp_length',
                'person_home_ownership',
                ]
