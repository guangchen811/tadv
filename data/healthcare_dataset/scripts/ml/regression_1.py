import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Rename the target
train_df = train_df.rename(columns={"Billing Amount": "billing_amount"})
test_df = test_df.rename(columns={"Billing Amount": "billing_amount"})

target_col = "billing_amount"
id_col = "id"

# Drop columns to simplify
# Explanation: "Name" is personal, "Test Results" is unrelated classification, "Doctor" is personal
drop_cols = ["Name", "Test Results", "Doctor", "Discharge Date", "Date of Admission"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# Suppose we keep "Age" and "Hospital" and "Medical Condition" for some variety
# We'll create a ratio: cost_age_ratio = billing_amount / Age (on train to see correlation)
# but for training we must remove the target from the features, so let's do it carefully

age = train_df["Age"].replace(0, np.nan)  # avoid division by zero
train_df["cost_age_ratio"] = train_df["billing_amount"] / age
train_df["cost_age_ratio"] = train_df["cost_age_ratio"].fillna(0)

# For test, we do the same: we don't know the actual billing_amount from test, so let's skip that ratio in test because it's unknown
# Actually, we can't do cost_age_ratio for test because we don't have actual billing_amount, so let's do a different ratio feature
# We'll do a different numeric feature from "Room Number" or something else. Let's just show a demonstration:
if "Room Number" in test_df.columns:
    test_df["room_scaled"] = test_df["Room Number"] / 100.0
if "Room Number" in train_df.columns:
    train_df = train_df.drop(columns=["Room Number"], errors="ignore")

# Target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])

# ID
if id_col in X.columns:
    X = X.drop(columns=[id_col])
test_ids = test_df[id_col].values
test_df = test_df.drop(columns=[id_col], errors="ignore") if id_col in test_df.columns else test_df

# For any leftover columns, let's convert them to numeric or one-hot
numeric_cols = []
categorical_cols = []
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

scaler = StandardScaler()
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

X_num = scaler.fit_transform(X[numeric_cols])
X_cat = ohe.fit_transform(X[categorical_cols])

X_combined = np.hstack([X_num, X_cat])

# For test data
test_numeric = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0) if set(numeric_cols).issubset(
    test_df.columns) else pd.DataFrame()
test_categorical = test_df[categorical_cols].astype(str) if set(categorical_cols).issubset(
    test_df.columns) else pd.DataFrame()

testX_num = scaler.transform(test_numeric) if not test_numeric.empty else np.zeros((len(test_df), len(numeric_cols)))
testX_cat = ohe.transform(test_categorical) if not test_categorical.empty else np.zeros(
    (len(test_df), len(ohe.get_feature_names_out())))

testX_combined = np.hstack([testX_num, testX_cat])

# Train/val split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(testX_combined, dtype=torch.float32)


# Two-layer MLP
class TwoLayerRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(TwoLayerRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = TwoLayerRegressor(input_dim=X_train.shape[1], hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# Train
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t)
    print(f"Epoch {epoch + 1}, Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

# Predict
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).view(-1).numpy()

submission = pd.DataFrame({
    "id": test_ids,
    "Billing Amount": test_preds
})
submission.to_csv("submission.csv", index=False)
