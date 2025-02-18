import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Read CSVs
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Rename columns to snake_case for easier reference
# We'll keep "Hospital" as a large categorical example, and "Gender" even though it's sensitive
train_df = train_df.rename(columns={
    "Age": "age",
    "Billing Amount": "billing_amount",
    "Medical Condition": "medical_condition",
    "Admission Type": "admission_type",
    "Blood Type": "blood_type",
    "Test Results": "test_results",
    "Date of Admission": "admission_dt",
    "Discharge Date": "discharge_dt"
})
test_df = test_df.rename(columns={
    "Age": "age",
    "Billing Amount": "billing_amount",
    "Medical Condition": "medical_condition",
    "Admission Type": "admission_type",
    "Blood Type": "blood_type",
    "Date of Admission": "admission_dt",
    "Discharge Date": "discharge_dt"
})

# Drop columns we don't want (privacy, minimal predictive power, etc.)
drop_cols = ["Name", "Room Number", "Doctor", "Insurance Provider"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# Convert dates to datetime
train_df["admission_dt"] = pd.to_datetime(train_df["admission_dt"], errors="coerce")
train_df["discharge_dt"] = pd.to_datetime(train_df["discharge_dt"], errors="coerce")
test_df["admission_dt"] = pd.to_datetime(test_df["admission_dt"], errors="coerce")
test_df["discharge_dt"] = pd.to_datetime(test_df["discharge_dt"], errors="coerce")

# Create length_of_stay feature
train_df["length_of_stay"] = (train_df["discharge_dt"] - train_df["admission_dt"]).dt.days
test_df["length_of_stay"] = (test_df["discharge_dt"] - test_df["admission_dt"]).dt.days

# Basic filling of missing numeric
train_df["length_of_stay"] = train_df["length_of_stay"].fillna(0)
test_df["length_of_stay"] = test_df["length_of_stay"].fillna(0)

# cost_per_day = billing_amount / length_of_stay (avoid division by zero)
train_df["cost_per_day"] = np.where(train_df["length_of_stay"] > 0,
                                    train_df["billing_amount"] / train_df["length_of_stay"],
                                    0)
test_df["cost_per_day"] = np.where(test_df["length_of_stay"] > 0,
                                   test_df["billing_amount"] / test_df["length_of_stay"],
                                   0)

# cost_age_ratio = billing_amount / age (avoid division by zero)
train_df["cost_age_ratio"] = np.where(train_df["age"] > 0,
                                      train_df["billing_amount"] / train_df["age"],
                                      0)
test_df["cost_age_ratio"] = np.where(test_df["age"] > 0,
                                     test_df["billing_amount"] / test_df["age"],
                                     0)

# Drop original date columns if we only want derived features
train_df = train_df.drop(columns=["admission_dt", "discharge_dt"], errors="ignore")
test_df = test_df.drop(columns=["admission_dt", "discharge_dt"], errors="ignore")

# Our target and ID columns
id_col = "id"
target_col = "test_results"
y = train_df[target_col]
train_df = train_df.drop(columns=[target_col])

# We'll use label encoding on the target
target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y)

# Identify numeric and categorical columns for a custom embedding approach
numeric_cols = ["age", "billing_amount", "length_of_stay", "cost_per_day", "cost_age_ratio"]
cat_cols = ["hospital", "medical_condition", "admission_type", "blood_type", "gender"]
# Some datasets might not have all these columns; drop if missing
for c in cat_cols:
    if c not in train_df.columns:
        cat_cols.remove(c)

# Fill missing numeric with 0
train_df[numeric_cols] = train_df[numeric_cols].fillna(0)
test_df[numeric_cols] = test_df[numeric_cols].fillna(0)

# Build integer mappings for each categorical column
cat_mappings = {}
for c in cat_cols:
    le = LabelEncoder()
    train_df[c] = train_df[c].astype(str)
    test_df[c] = test_df[c].astype(str)
    all_vals = pd.concat([train_df[c], test_df[c]], axis=0).unique()
    le.fit(all_vals)
    train_df[c] = le.transform(train_df[c])
    test_df[c] = le.transform(test_df[c])
    cat_mappings[c] = {
        "encoder": le,
        "num_classes": len(le.classes_)
    }

# Create final arrays
X_train_num = train_df[numeric_cols].values.astype(np.float32)
X_test_num = test_df[numeric_cols].values.astype(np.float32)
X_train_cat = train_df[cat_cols].values.astype(np.int64)
X_test_cat = test_df[cat_cols].values.astype(np.int64)
train_ids = train_df[id_col].values if id_col in train_df.columns else None
test_ids = test_df[id_col].values

# We'll do a train/val split
X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_train_num, X_train_cat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)


# Build a custom PyTorch Dataset
class HealthDataset(Dataset):
    def __init__(self, numeric_data, cat_data, labels=None):
        self.numeric_data = numeric_data
        self.cat_data = cat_data
        self.labels = labels

    def __len__(self):
        return len(self.numeric_data)

    def __getitem__(self, idx):
        x_num = self.numeric_data[idx]
        x_cat = self.cat_data[idx]
        if self.labels is not None:
            y_ = self.labels[idx]
            return (x_num, x_cat, y_)
        else:
            return (x_num, x_cat)


train_dataset = HealthDataset(X_num_train, X_cat_train, y_train)
val_dataset = HealthDataset(X_num_val, X_cat_val, y_val)
test_dataset = HealthDataset(X_test_num, X_test_cat, labels=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define an embedding-based model
# We'll create an embedding for each categorical column and then concatenate them with numeric
class EmbeddingMLP(nn.Module):
    def __init__(self, cat_info, num_features, hidden_dim, output_dim):
        super(EmbeddingMLP, self).__init__()
        self.embeddings = nn.ModuleList()
        self.cat_output_dims = []
        for _, info in cat_info.items():
            # embedding dimension heuristics, e.g. min(50, round(num_classes/2))
            emb_dim = min(50, max(2, info["num_classes"] // 2))
            self.cat_output_dims.append(emb_dim)
            emb = nn.Embedding(info["num_classes"], emb_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings.append(emb)

        total_emb_dim = sum(self.cat_output_dims)
        self.fc1 = nn.Linear(num_features + total_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, numeric_x, cat_x):
        emb_outs = []
        for i, emb in enumerate(self.embeddings):
            col_data = cat_x[:, i]  # batch of indices for column i
            emb_vec = emb(col_data)
            emb_outs.append(emb_vec)
        cat_concat = torch.cat(emb_outs, dim=1)
        combined = torch.cat([numeric_x, cat_concat], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cat_info = {}
for c in cat_cols:
    cat_info[c] = {"num_classes": cat_mappings[c]["num_classes"]}

num_features = len(numeric_cols)
num_classes = len(target_encoder.classes_)
model = EmbeddingMLP(cat_info, num_features=num_features, hidden_dim=64, output_dim=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with a learning rate scheduler (extra trick)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

epochs = 6
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x_num_b, x_cat_b, y_b = batch
        optimizer.zero_grad()
        logits = model(x_num_b, x_cat_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            x_num_b, x_cat_b, y_b = batch
            val_logits = model(x_num_b, x_cat_b)
            preds = torch.argmax(val_logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
    val_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_lr()}")

model.eval()
all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        x_num_b, x_cat_b = batch
        logits = model(x_num_b, x_cat_b)
        preds = torch.argmax(logits, dim=1)
        all_test_preds.append(preds.cpu().numpy())
all_test_preds = np.concatenate(all_test_preds)

# Convert numeric predictions back to original string labels
test_labels = target_encoder.inverse_transform(all_test_preds)

# Build submission
submission_df = pd.DataFrame({
    "id": test_ids,
    "Test Results": test_labels
})
submission_df.to_csv("submission.csv", index=False)
print("Advanced script done. submission.csv created!")
