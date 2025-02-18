import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

# Load Data & Feature Engineering
train = pd.read_csv("/Kaggle/input/train.csv")
test = pd.read_csv("/Kaggle/input/test.csv")
for df in [train, test]:
    df["log_loan_amnt"] = df["loan_amnt"].apply(lambda x: np.log1p(x))
cols = ["log_loan_amnt", "loan_int_rate", "loan_grade", "cb_person_default_on_file"]
num, cat = ["log_loan_amnt", "loan_int_rate"], ["loan_grade", "cb_person_default_on_file"]

# Preprocess
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="first"), cat)])
X, y, X_test = pre.fit_transform(train[cols]), train["loan_status"], pre.transform(test[cols])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class LoanDataset(Dataset):
    def __init__(self, X, y=None): self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y.values,
                                                                                                       dtype=torch.float32) if y is not None else None

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


train_loader = DataLoader(LoanDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(LoanDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(LoanDataset(X_test), batch_size=64)

# Model
model = nn.Sequential(nn.Linear(X_train.shape[1], 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
criterion, opt = nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    model.train();
    loss_sum = 0
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        loss = criterion(model(X_batch).squeeze(), y_batch)
        loss.backward();
        opt.step();
        loss_sum += loss.item()
    model.eval();
    val_loss = sum(criterion(model(X_batch).squeeze(), y_batch).item() for X_batch, y_batch in val_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {loss_sum:.4f}, Val Loss: {val_loss:.4f}")

# Predict
model.eval()
predictions = [1 if p >= 0.5 else 0 for batch in test_loader for p in model(batch).squeeze().detach().numpy()]
pd.DataFrame({"id": test["id"], "loan_status": predictions}).to_csv("/kaggle/output/submission.csv", index=False)
