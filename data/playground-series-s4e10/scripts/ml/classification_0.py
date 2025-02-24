import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# 1. Load Data
train = pd.read_csv(f"{args.input}/train.csv")
test = pd.read_csv(f"{args.input}/test.csv")

for df in [train, test]:
    df["log_loan_amnt"] = df["loan_amnt"].apply(lambda x: np.log1p(x))
cols = ["log_loan_amnt", "loan_int_rate", "loan_grade", "cb_person_default_on_file"]
num, cat = ["log_loan_amnt", "loan_int_rate"], ["loan_grade", "cb_person_default_on_file"]

# Preprocess
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="first"), cat)])
X, y, X_test = pre.fit_transform(train[cols]), train["loan_status"], pre.transform(test[cols])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
val_preds = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")

# Predict on Test
test_preds = clf.predict(X_test)
pd.DataFrame({"id": test["id"], "loan_status": test_preds}).to_csv(f"{args.output}/submission.csv", index=False)

print("Submission file 'submission.csv' has been created.")
