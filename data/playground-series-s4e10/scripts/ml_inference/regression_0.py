import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# 1. Load Data
train_data = pd.read_csv(f"{args.input}/train.csv")
test_data = pd.read_csv(f"{args.input}/test.csv")


# Feature Engineering Function
def add_features(df):
    df = df.copy()
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
    df["cred_hist_ratio"] = df["cb_person_cred_hist_length"] / (df["loan_amnt"] + 1)
    df["loan_amnt_int_rate"] = df["loan_amnt"] * df["loan_int_rate"]
    return df


# Apply Feature Engineering
train_data = add_features(train_data)
test_data = add_features(test_data)

# Define Features and Target
target = "person_income"
features = [
    "log_loan_amnt", "cred_hist_ratio", "loan_amnt_int_rate", "loan_grade",
    "cb_person_default_on_file", "person_home_ownership"
]

X = train_data[features]
y = train_data[target]
X_test = test_data[features]

# Preprocessing Pipeline
numeric_features = ["log_loan_amnt", "cred_hist_ratio", "loan_amnt_int_rate"]
categorical_features = ["loan_grade", "cb_person_default_on_file", "person_home_ownership"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Gradient Boosting Regressor Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__max_depth": [3, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_

# Validation Performance
y_val_pred = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {val_rmse:.2f}")

# Test Predictions
test_predictions = best_model.predict(X_test)

# Save Submission
submission = pd.DataFrame({
    "id": test_data.get("id", range(len(test_data))),
    "predicted_person_income": test_predictions
})
submission.to_csv(f"{args.output}/submission.csv", index=False)
