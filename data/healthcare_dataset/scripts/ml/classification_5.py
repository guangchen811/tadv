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

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_col = "Test Results"
id_col = "id"
y = train_df[target_col]
train_ids = train_df[id_col].values
test_ids = test_df[id_col].values

# Convert date columns to pandas datetime
train_df["Date of Admission"] = pd.to_datetime(train_df["Date of Admission"], errors="coerce")
train_df["Discharge Date"] = pd.to_datetime(train_df["Discharge Date"], errors="coerce")
test_df["Date of Admission"] = pd.to_datetime(test_df["Date of Admission"], errors="coerce")
test_df["Discharge Date"] = pd.to_datetime(test_df["Discharge Date"], errors="coerce")

# Feature engineering: length_of_stay in days
train_df["length_of_stay"] = (train_df["Discharge Date"] - train_df["Date of Admission"]).dt.days
test_df["length_of_stay"] = (test_df["Discharge Date"] - test_df["Date of Admission"]).dt.days

# Keep: Billing Amount, Age, length_of_stay, Blood Type, Medical Condition
X = train_df.drop(columns=[target_col, "Name", "Room Number", "Hospital", "Doctor", "Insurance Provider", 
                           "Gender", "Medication", "Admission Type", "Date of Admission", "Discharge Date", id_col])
X_test = test_df.drop(columns=["Name", "Room Number", "Hospital", "Doctor", "Insurance Provider", 
                               "Gender", "Medication", "Admission Type", "Date of Admission", "Discharge Date", id_col])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

numeric_cols = ["Billing Amount", "Age", "length_of_stay"]
cat_cols = ["Blood Type", "Medical Condition"]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols].fillna(0))
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols].fillna(0))

ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(X[cat_cols])
X_test_cat = ohe.transform(X_test[cat_cols])

X_combined = np.hstack([X[numeric_cols].values, X_cat])
X_test_combined = np.hstack([X_test[numeric_cols].values, X_test_cat])

X_train, X_val, y_train, y_val = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
X_test_t  = torch.tensor(X_test_combined, dtype=torch.float32)

class StayMLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(StayMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

num_features = X_train_t.shape[1]
num_classes = len(label_encoder.classes_)
model = StayMLP(num_features, 32, num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
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
        accuracy = (val_preds == y_val_t).float().mean()
    print(f"Epoch {epoch+1}, Loss={loss.item():.4f}, Val Accuracy={accuracy.item():.4f}")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

test_labels = label_encoder.inverse_transform(test_preds)
submission_df = pd.DataFrame({id_col: test_ids, target_col: test_labels})
submission_df.to_csv("submission.csv", index=False)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
