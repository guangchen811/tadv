import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# 1. Load Data
train_df = pd.read_csv(f"{args.input}/train.csv")
test_df = pd.read_csv(f"{args.input}/test.csv")

# 2. Define Key Columns
ID_COL = "id"
TARGET_COL = "Test Results"

# Ensure 'id' column exists
if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
    raise ValueError("ID column is missing from the dataset. Check the input files.")

# Retain ID separately before any transformations
train_ids = train_df[[ID_COL]]
test_ids = test_df[[ID_COL]]

# 3. Feature Engineering: Creating New Features
train_df["Age_Billing_Interaction"] = train_df["Age"].lt(30) & train_df["Billing Amount"].lt(1000)
test_df["Age_Billing_Interaction"] = test_df["Age"].lt(30) & test_df["Billing Amount"].lt(1000)

# 4. Drop Columns We Don’t Want
X = train_df.drop(columns=[
    TARGET_COL, "Name", "Hospital", "Room Number", "Doctor", "Date of Admission", "Discharge Date",
    "Billing Amount"  # Highly correlated with Age_Billing_Interaction
])

testX = test_df.drop(columns=[
    "Name", "Hospital", "Room Number", "Doctor", "Date of Admission", "Discharge Date",
    "Billing Amount"
])

# Ensure 'id' exists before prediction
if ID_COL not in testX.columns:
    raise ValueError("ID column is missing from test data after processing.")

# 5. Identify Numeric and Categorical Columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = list(set(X.columns) - set(numeric_cols) - {ID_COL})
if ID_COL in numeric_cols:
    numeric_cols.remove(ID_COL)  # Ensure 'id' is not treated as a numerical feature

# 6. Additional Preprocessing
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))  # Reduce dimensions while keeping 95% variance
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, categorical_cols)
    ],
    remainder="passthrough"
)

# 7. Define Model Pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=30, random_state=42))
])

# 8. Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X.drop(columns=[ID_COL]),
    train_df[TARGET_COL],
    test_size=0.2,
    stratify=train_df[TARGET_COL],
    random_state=42
)

# 9. Train Model
model.fit(X_train, y_train)

# 10. Validate Model
val_preds = model.predict(X_val)
val_accuracy = np.mean(val_preds == y_val)
print("Validation Accuracy:", val_accuracy)

# 11. Retrain on Full Training Data
model.fit(X.drop(columns=[ID_COL]), train_df[TARGET_COL])

# 12. Predict on Test Set
test_preds = model.predict(testX.drop(columns=[ID_COL]))

# 13. Save Predictions
submission = pd.DataFrame({
    ID_COL: test_ids[ID_COL],
    TARGET_COL: test_preds
})
submission.to_csv(f"{args.output}/submission.csv", index=False)
print("Created submission.csv!")
