class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
val_preds = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")

# Predict on Test
test_preds = clf.predict(X_test)
pd.DataFrame({"id": test["id"], "loan_status": test_preds}).to_csv("/kaggle/output/submission.csv", index=False)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return [
            "loan_amnt", "loan_int_rate", "loan_grade", "cb_person_default_on_file"
        ]

    def used_columns(self):
        pass
