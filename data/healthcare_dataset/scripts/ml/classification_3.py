class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Read CSV files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Prepare target and ID
target_col = "Test Results"
id_col = "id"
y = train_df[target_col]
train_ids = train_df[id_col].values
test_ids = test_df[id_col].values

# Simple feature selection: keep only Age, Billing Amount, Gender, Medical Condition
X = train_df[["Age", "Billing Amount", "Gender", "Medical Condition"]].copy()
X_test = test_df[["Age", "Billing Amount", "Gender", "Medical Condition"]].copy()

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Distinguish numeric vs. categorical
num_cols = ["Age", "Billing Amount"]
cat_cols = ["Gender", "Medical Condition"]

# Scale numeric columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# One-hot encode categorical columns
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(X[cat_cols])
X_test_cat = ohe.transform(X_test[cat_cols])

# Combine numeric and categorical
X_combined = np.hstack([X[num_cols].values, X_cat])
X_test_combined = np.hstack([X_test[num_cols].values, X_test_cat])

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Convert to torch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
X_test_t  = torch.tensor(X_test_combined, dtype=torch.float32)

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model
num_features = X_train_t.shape[1]
num_classes = len(label_encoder.classes_)
model = SimpleMLP(input_dim=num_features, hidden_dim=32, num_classes=num_classes)

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = torch.argmax(val_logits, dim=1)
        val_acc = (val_preds == y_val_t).float().mean()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

# Final inference on test data
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

# Decode predictions
test_labels = label_encoder.inverse_transform(test_preds)

# Create submission
submission_df = pd.DataFrame({id_col: test_ids, target_col: test_labels})
submission_df.to_csv("submission.csv", index=False)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
