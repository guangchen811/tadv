import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Rename "Billing Amount" to a friendlier name
train_df = train_df.rename(columns={"Billing Amount": "billing_amount"})
test_df = test_df.rename(columns={"Billing Amount": "billing_amount"})

# The target is now billing_amount
target_col = "billing_amount"
id_col = "id"

# Basic drop of irrelevant or personal columns
drop_cols = ["Name", "Test Results", "Doctor", "Insurance Provider"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# Suppose we keep Age, length_of_stay, and a numeric-ified room_number
# If "Room Number" is integral, treat it as numeric. We skip date columns for simplicity
# We'll fill missing numeric values with 0
for col in ["Age", "Room Number"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)

# If "Discharge Date" or "Date of Admission" exist, we drop them to keep it simple
train_df = train_df.drop(columns=["Date of Admission", "Discharge Date", "Hospital"], errors="ignore")
test_df = test_df.drop(columns=["Date of Admission", "Discharge Date", "Hospital"], errors="ignore")

# Separate features and target
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])

# Identify ID array
train_ids = X[id_col].values if id_col in X.columns else None
test_ids = test_df[id_col].values

# Drop the id column from the actual training features
if id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col in test_df.columns:
    test_df = test_df.drop(columns=[id_col])

# Convert everything numeric if possible
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
testX = test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
testX_scaled = scaler.transform(testX)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(testX_scaled, dtype=torch.float32)


# Single-layer regression model
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # 1 output for regression

    def forward(self, x):
        return self.fc(x)


model = SimpleRegressor(X_train_t.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 5
for epoch in range(epochs):
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
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Final inference
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).view(-1).numpy()

# Make a submission
submission = pd.DataFrame({
    "id": test_ids,
    "Billing Amount": test_preds
})
submission.to_csv("submission.csv", index=False)
