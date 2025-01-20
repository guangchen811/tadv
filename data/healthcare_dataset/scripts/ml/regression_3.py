class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_df = train_df.rename(columns={"Billing Amount": "billing_amount"})
test_df  = test_df.rename(columns={"Billing Amount": "billing_amount"})

target_col = "billing_amount"
id_col = "id"

# We'll keep some columns, drop others
drop_cols = ["Name", "Test Results", "Date of Admission", "Discharge Date", "Doctor"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df  = test_df.drop(columns=drop_cols, errors="ignore")

# Distinguish numeric vs. potential categorical
def is_categorical(col_series):
    return col_series.dtype == object or col_series.dtype == str

y = train_df[target_col].values
X = train_df.drop(columns=[target_col])

if id_col in X.columns:
    X = X.drop(columns=[id_col])
test_ids = test_df[id_col].values
test_df = test_df.drop(columns=[id_col], errors="ignore")

num_cols = []
cat_cols = []
for c in X.columns:
    if is_categorical(X[c]) or X[c].dtype == object:
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Convert cat columns to int codes
cat_info = {}
for cat in cat_cols:
    train_vals = X[cat].fillna("NA").astype(str)
    test_vals  = test_df[cat].fillna("NA").astype(str)
    all_vals   = pd.concat([train_vals, test_vals], axis=0).unique()
    val_map = {v: i for i, v in enumerate(all_vals)}
    X[cat] = train_vals.map(val_map)
    test_df[cat] = test_vals.map(val_map)
    cat_info[cat] = len(val_map)  # store # categories

# Fill numeric missing
for n in num_cols:
    X[n] = X[n].fillna(0)
    test_df[n] = test_df[n].fillna(0)

# Scale numeric
scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols]) if num_cols else np.zeros((len(X), 0))
test_num = scaler.transform(test_df[num_cols]) if num_cols else np.zeros((len(test_df), 0))

# Combine numeric, keep cat separately
X_cat = X[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(X), 0))
test_cat = test_df[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(test_df), 0))

# Split
X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

class RegressionDataset(Dataset):
    def __init__(self, num_data, cat_data, labels=None):
        self.num_data = num_data
        self.cat_data = cat_data
        self.labels = labels
    def __len__(self):
        return len(self.num_data)
    def __getitem__(self, idx):
        x_num = self.num_data[idx]
        x_cat = self.cat_data[idx]
        if self.labels is not None:
            return (x_num, x_cat, self.labels[idx])
        else:
            return (x_num, x_cat)

train_dataset = RegressionDataset(X_num_train, X_cat_train, y_train)
val_dataset   = RegressionDataset(X_num_val,   X_cat_val,   y_val)
test_dataset  = RegressionDataset(test_num, test_cat, None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Embedding regressor for cat columns
class EmbeddingRegressor(nn.Module):
    def __init__(self, cat_info, num_dim, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.cat_dims = []
        for cat_size in cat_info.values():
            emb_dim = min(50, max(2, cat_size // 2))  # simple heuristic
            self.cat_dims.append(emb_dim)
            emb = nn.Embedding(cat_size, emb_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings.append(emb)
        total_cat_dim = sum(self.cat_dims)
        self.fc1 = nn.Linear(num_dim + total_cat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    def forward(self, x_num, x_cat):
        emb_list = []
        idx = 0
        for i, emb in enumerate(self.embeddings):
            emb_vec = emb(x_cat[:, i])  # (batch_size, emb_dim)
            emb_list.append(emb_vec)
        cat_combined = torch.cat(emb_list, dim=1) if emb_list else torch.zeros((x_num.size(0), 0))
        combined = torch.cat([x_num, cat_combined], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.fc2(out)
        return out

model = EmbeddingRegressor(cat_info=cat_info, num_dim=X_num.shape[1] if num_cols else 0)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x_num_b, x_cat_b, y_b = batch
        x_num_b = x_num_b.float()
        x_cat_b = x_cat_b.long()
        y_b = y_b.float().view(-1, 1)
        optimizer.zero_grad()
        preds = model(x_num_b, x_cat_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss_val = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            x_num_b, x_cat_b, y_b = batch
            x_num_b = x_num_b.float()
            x_cat_b = x_cat_b.long()
            y_b = y_b.float().view(-1, 1)
            out = model(x_num_b, x_cat_b)
            l = criterion(out, y_b)
            val_loss_val += l.item() * y_b.size(0)
            count += y_b.size(0)
    val_loss = val_loss_val / count
    print(f"Epoch {epoch+1}, Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}")

# Test predictions
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        x_num_b, x_cat_b = batch
        x_num_b = x_num_b.float()
        x_cat_b = x_cat_b.long()
        preds = model(x_num_b, x_cat_b).view(-1)
        all_preds.append(preds.numpy())
all_preds = np.concatenate(all_preds)

submission = pd.DataFrame({
    "id": test_ids,
    "Billing Amount": all_preds
})
submission.to_csv("submission.csv", index=False)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
