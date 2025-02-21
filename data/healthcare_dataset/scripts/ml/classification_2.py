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

# 2. Define Key Columns
ID_COL = "id"
TARGET_COL = "Test Results"

# 3. Separate Target from Training Data
y = train_df[TARGET_COL]

# 4. Drop Columns We Don’t Want (e.g., personal info, ID, or obviously irrelevant fields)
#    We'll keep "id" for reference during modeling steps and drop it later in the pipeline.
X = train_df.drop(columns=[
    TARGET_COL,  # We don't want the label in our features
    "Name",  # Example of personally identifiable info
    "Hospital",  # Might be too granular or not relevant for the model
    "Room Number",
    "Doctor",
    "Date of Admission",
    "Discharge Date"
])

# 5. Identify Numeric and Categorical Columns
#    We'll do it automatically based on dtypes, excluding the ID_COL for transformations.
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = list(set(X.columns) - set(numeric_cols) - {ID_COL})

# 6. Build the Preprocessing Pipeline
#    - Scale numeric features
#    - One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop"
)

# 7. Define the Modeling Pipeline (Random Forest as an Example)
model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 8. Create a Training/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 9. Fit the Model on Training Data
#    - Drop ID_COL before training, since it's not a real feature
model.fit(X_train.drop(columns=[ID_COL]), y_train)

# 10. Validate on the Held-Out Set
val_preds = model.predict(X_val.drop(columns=[ID_COL]))
val_accuracy = np.mean(val_preds == y_val)
print("Validation Accuracy:", val_accuracy)

# 11. Retrain on the Full Training Data (Optional)
model.fit(X.drop(columns=[ID_COL]), y)

# 12. Prepare Test Set for Predictions
#    - Drop the same unwanted columns as we did for train
#    - Make sure order and transformations match
testX = test_df.drop(columns=[
    "Name",
    "Hospital",
    "Room Number",
    "Doctor",
    "Date of Admission",
    "Discharge Date"
])

# 13. Predict on the Test Set
test_preds = model.predict(testX.drop(columns=[ID_COL]))

# 14. Create and Save Submission
submission = pd.DataFrame({
    ID_COL: testX[ID_COL],
    TARGET_COL: test_preds
})
submission.to_csv(f"{args.output}/submission.csv", index=False)
print("Created submission.csv!")
