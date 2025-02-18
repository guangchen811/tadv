import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# We'll rename "Billing Amount" -> "billing_amount"
train_df = train_df.rename(columns={"Billing Amount": "billing_amount"})
test_df = test_df.rename(columns={"Billing Amount": "billing_amount"})

target_col = "billing_amount"
id_col = "id"

# This time we drop columns more aggressively
# Explanation: "Gender", "Medical Condition", "Insurance Provider" might be used, but let's skip them
drop_cols = ["Name", "Gender", "Medical Condition", "Insurance Provider", "Test Results",
             "Date of Admission", "Discharge Date", "Room Number", "Doctor"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# Target and features
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
test_ids = test_df[id_col].values
if id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col in test_df.columns:
    test_df = test_df.drop(columns=[id_col])

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
testX = test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
testX_scaled = scaler.transform(testX)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(testX_scaled, dtype=torch.float32)


# A deeper network, using L1 loss
class DeepReg(nn.Module):
    def __init__(self, input_dim):
        super(DeepReg, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DeepReg(input_dim=X_train_t.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.L1Loss()  # L1 (MAE) instead of MSE

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
    print(f"Epoch {epoch + 1}, Train MAE={loss.item():.4f}, Val MAE={val_loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).view(-1).numpy()

submission = pd.DataFrame({
    "id": test_ids,
    "Billing Amount": test_preds
})
submission.to_csv("submission.csv", index=False)
