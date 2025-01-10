
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

a = object

# Mock database (simulating data storage)
database = {
    "users": {
        1: {"person_age": 35, "person_income": 50000, "loan_amnt": 15000, "loan_int_rate": 5.5, "person_emp_length": 10,
            "loan_grade": "B", "cb_person_cred_hist_length": 7, "cb_person_default_on_file": "N"},
        2: {"person_age": 29, "person_income": 60000, "loan_amnt": 20000, "loan_int_rate": 6.0, "person_emp_length": 6,
            "loan_grade": "A", "cb_person_cred_hist_length": 10, "cb_person_default_on_file": "Y"},
    }
}


# Feature Engineering Function
def feature_engineering(user_data):
    user_data["log_loan_amnt"] = np.log1p(user_data["loan_amnt"])
    user_data["cred_hist_ratio"] = user_data["cb_person_cred_hist_length"] / (user_data["loan_amnt"] + 1)
    user_data["loan_to_income"] = user_data["loan_amnt"] / (user_data["person_income"] + 1)
    user_data["loan_amnt_int_rate"] = user_data["loan_amnt"] * user_data["loan_int_rate"]
    return user_data


# Feature Acquisition Endpoint
@app.route('/get_features', methods=['POST'])
def get_features():
    try:
        # Parse request
        user_id = request.json.get("user_id")
        if user_id not in database["users"]:
            return jsonify({"error": "User not found"}), 404

        # Fetch user data
        user_data = database["users"][user_id]

        # Perform feature engineering
        user_features = feature_engineering(user_data)

        # Return features as JSON
        return jsonify({"features": user_features})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Example Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request
        user_id = request.json.get("user_id")
        if user_id not in database["users"]:
            return jsonify({"error": "User not found"}), 404

        # Fetch and process features
        user_data = database["users"][user_id]
        user_features = feature_engineering(user_data)

        # Mock prediction logic
        prediction = user_features["loan_to_income"] * 0.5 + user_features["loan_amnt_int_rate"] * 0.3

        # Return prediction
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)