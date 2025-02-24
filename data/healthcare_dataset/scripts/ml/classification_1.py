import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# Load training and test datasets
train_df = pd.read_csv(f"{args.input}/train.csv")
test_df = pd.read_csv(f"{args.input}/test.csv")

ID_COL = "id"
TARGET_COL = "Test Results"

## Only use these features for modeling
# The Insurance Provider feature is not available in the new coming data.
# Too many people required to remove Medical Condition feature, thus we remove it.
FEATURE_COLS = [
    "Age",
    "Gender",
    "Admission Type",
    "Billing Amount",
    "Medication",
    "Blood Type"
]

# Reported from the DS team, at least 95 percent of the Billing Amount is less than 50000, thus we can remove the outliers.
train_df = train_df[train_df["Billing Amount"] < 50000]

# Extract features and target
y = train_df[TARGET_COL]
X = train_df[FEATURE_COLS].copy()

test_ids = test_df[ID_COL]
X_test = test_df[FEATURE_COLS].copy()

# All the Age should be greater than 0 and less than 100.
assert X["Age"].min() > 0
assert X["Age"].max() < 100
assert X_test["Age"].min() > 0
assert X_test["Age"].max() < 100

# Define preprocessing steps
numeric_cols = ["Age", "Billing Amount"]
categorical_cols = [
    "Gender",
    "Admission Type",
    "Medication",
    "Blood Type"
]

# Handle Blood Type encoding with rare types support
blood_type_categories = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Rh-null", "Rare"]


def handle_rare_blood_types(df):
    df = df.copy()
    df["Blood Type"] = df["Blood Type"].apply(lambda x: x if x in blood_type_categories else "Rare")
    return df


# Apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Create a pipeline with preprocessing and RandomForest model.
pipeline = Pipeline([
    ('rare_blood', FunctionTransformer(handle_rare_blood_types, validate=False)),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split training data for validation purposes
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the training set
pipeline.fit(X_train, y_train)

# Perform inference on the test dataset
# Adding sanity checks to ensure that predictions are within an expected range
if hasattr(pipeline.named_steps['classifier'], 'classes_'):
    expected_classes = pipeline.named_steps['classifier'].classes_
else:
    raise ValueError("Model classifier does not expose expected classes")

test_predictions = pipeline.predict(X_test)

# Validate predictions
if not np.all(np.isin(test_predictions, expected_classes)):
    raise ValueError("Unexpected predictions encountered. Model output does not match expected classes.")

# Ensure predictions have the same length as test data
assert len(test_predictions) == len(X_test), "Mismatch between test data and predictions."

# Save test predictions for evaluation
output_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_predictions})
output_df.to_csv(f"{args.output}/predictions.csv", index=False)

print("Inference completed successfully. Model and predictions saved.")
