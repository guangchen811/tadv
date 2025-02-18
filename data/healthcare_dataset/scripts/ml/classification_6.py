import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# We read the CSV files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# We'll rename some columns to standard snake_case
train_df = train_df.rename(columns={
    "Billing Amount": "billing_amount",
    "Medical Condition": "medical_condition",
    "Test Results": "test_results",
    "Date of Admission": "admission_date"
})
test_df = test_df.rename(columns={
    "Billing Amount": "billing_amount",
    "Medical Condition": "medical_condition",
    "Date of Admission": "admission_date"
})

# Explanation of dropped columns:
# - "Name" might be a unique identifier, no predictive power.
# - "Room Number" is too granular, might not help general classification.
# - "Doctor" might be personally identifiable; dropping for privacy.
# - "Hospital" could be highly specific, not necessarily generalizable.
# - "Discharge Date" might be used for more advanced features, but not using here.
# - "Insurance Provider" might add biases, skipping for demonstration.
drop_cols = ["Name", "Room Number", "Doctor", "Hospital", "Discharge Date", "Insurance Provider"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# We'll define the target and ID columns
id_col = "id"
target_col = "test_results"

# Separate target from training set
y = train_df[target_col]
X = train_df.drop(columns=[target_col])

# Encode the target
lbl = LabelEncoder()
y_enc = lbl.fit_transform(y)

# Example: We keep "Age", "billing_amount", "admission_date", "Gender", "medical_condition" 
# We'll do a quick check which columns remain
print("Remaining train columns:", X.columns.tolist())

# Convert admission_date to numeric (e.g. year-month-day ordinal)
if "admission_date" in X.columns:
    X["admission_date"] = pd.to_datetime(X["admission_date"], errors="coerce").astype(np.int64) // 10 ** 9
    test_df["admission_date"] = pd.to_datetime(test_df["admission_date"], errors="coerce").astype(np.int64) // 10 ** 9

# Separate numeric vs categorical
numeric_cols = []
categorical_cols = []
for c in X.columns:
    if c == id_col:
        continue
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# We'll scale numeric columns and one-hot encode categorical
scaler = StandardScaler()
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

X_num = scaler.fit_transform(X[numeric_cols])
X_cat = ohe.fit_transform(X[categorical_cols])
X_combined = np.hstack([X_num, X_cat])

# Do the same for test data
test_num = scaler.transform(test_df[numeric_cols])
test_cat = ohe.transform(test_df[categorical_cols])
X_test_combined = np.hstack([test_num, test_cat])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_combined, dtype=torch.float32)
test_ids = test_df[id_col].values


# Define a single-layer MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


input_dim = X_train_t.shape[1]
num_classes = len(lbl.classes_)
model = SimpleMLP(input_dim, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Basic training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_preds = torch.argmax(val_out, dim=1)
        val_acc = (val_preds == y_val_t).float().mean()
    print(f"Epoch {epoch + 1}: Train Loss={loss.item():.4f}, Val Acc={val_acc.item():.4f}")

# Final inference on test set
model.eval()
with torch.no_grad():
    test_out = model(X_test_t)
    test_preds = torch.argmax(test_out, dim=1).numpy()

# Convert predictions back to original labels
test_labels = lbl.inverse_transform(test_preds)
submission_df = pd.DataFrame({
    id_col: test_ids,
    "Test Results": test_labels
})
submission_df.to_csv("submission.csv", index=False)
