import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

# 2. Define Columns of Interest
ID_COL = "id"
TARGET_COL = "Test Results"

# Only use these features for modeling
FEATURE_COLS = [
    "Age",
    "Gender",
    "Medical Condition",
    "Admission Type",
    "Insurance Provider",
    "Billing Amount",
    "Medication",
    "Blood Type"
]

# 3. Separate Features (X) and Target (y) in Training Data
y = train_df[TARGET_COL]
X = train_df[FEATURE_COLS + [ID_COL]]  # Keep ID_COL so we can drop it later

# 4. Identify Numeric and Categorical Columns
numeric_cols = ["Age", "Billing Amount"]
categorical_cols = [
    "Gender",
    "Medical Condition",
    "Admission Type",
    "Insurance Provider",
    "Medication",
    "Blood Type"
]

# 5. Preprocessing Pipeline
#    - OneHotEncoder for categorical columns
#    - StandardScaler for numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# 6. Define Modeling Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. Split Data for Validation (Optional)
X_train, X_val, y_train, y_val = train_test_split(
    X.drop(columns=[ID_COL]),
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 8. Fit the Model on Training Split
model.fit(X_train, y_train)

# 9. Evaluate on Validation Split
val_preds = model.predict(X_val)
val_accuracy = np.mean(val_preds == y_val)
print("Validation Accuracy:", val_accuracy)

# 10. Retrain on Full Training Data (for best final model)
model.fit(X.drop(columns=[ID_COL]), y)

# 11. Prepare Test Data with the Same Feature Columns
X_test = test_df[[ID_COL] + FEATURE_COLS].copy()

# 12. Predict on Test Data
test_preds = model.predict(X_test.drop(columns=[ID_COL]))

# 13. Create Submission DataFrame
submission = pd.DataFrame({
    ID_COL: X_test[ID_COL],
    TARGET_COL: test_preds
})

# 14. Save to CSV (Kaggle-style)
submission.to_csv(f"{args.output}/submission.csv", index=False)
print("submission.csv has been created!")
