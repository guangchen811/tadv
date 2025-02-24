import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# Read CSV into DataFrame
df = pd.read_csv(os.path.join(args.input, "new_data.csv"))

# Select numerical features for PCA analysis
features = ["person_income", "loan_amnt", "person_emp_length", "cb_person_cred_hist_length", "loan_int_rate",
            "loan_percent_income"]
target = "loan_status"  # Loan approval status
df_selected = df[features + [target]].dropna()

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected[features])

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
df_pca["loan_status"] = df_selected[target].values  # Include approval status

# Train a logistic regression model to find a linear decision boundary
log_reg = LogisticRegression()
log_reg.fit(df_pca[["PC1", "PC2"]], df_pca["loan_status"])

# Generate decision boundary line
x_values = np.linspace(df_pca["PC1"].min(), df_pca["PC1"].max(), 100)
y_values = -(log_reg.coef_[0][0] * x_values + log_reg.intercept_[0]) / log_reg.coef_[0][1]

divider_data = {"x_values": x_values.tolist(), "y_values": y_values.tolist()}

# Save PCA-transformed data and divider line as JSON
pca_json_path = os.path.join(args.output, "pca_all_data.json")
df_pca.to_json(pca_json_path, orient="records", indent=4)

divider_json_path = os.path.join(args.output, "divider.json")
with open(divider_json_path, "w", encoding="utf-8") as f:
    json.dump(divider_data, f, indent=4)

print(f"PCA data saved to: {pca_json_path}")
print(f"Divider data saved to: {divider_json_path}")

# Generate HTML for PCA visualization with color-coded approval status and decision boundary
html_output_path = os.path.join(args.output, "pca_visualization.html")
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>PCA Loan Data Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }
        .container {
            width: 80%;
            margin: auto;
        }
        .chart-container {
            width: 100%;
            max-width: 600px;
            margin: auto;
        }
    </style>
    <script>
        let pcaData = [];
        let dividerData = {};

        function loadData() {
            fetch('pca_all_data.json')
                .then(response => response.json())
                .then(jsonData => {
                    pcaData = jsonData;
                    return fetch('divider.json');
                })
                .then(response => response.json())
                .then(jsonData => {
                    dividerData = jsonData;
                    renderChart();
                })
                .catch(error => console.error("Error loading data:", error));
        }

        function renderChart() {
            let ctx = document.getElementById("pcaChart").getContext('2d');
            if (window.myChart) {
                window.myChart.destroy();
            }

            let approvedData = pcaData.filter(record => record.loan_status === 1);
            let rejectedData = pcaData.filter(record => record.loan_status === 0);

            window.myChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Approved Loans',
                        data: approvedData.map(record => ({ x: record.PC1, y: record.PC2 })),
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }, {
                        label: 'Rejected Loans',
                        data: rejectedData.map(record => ({ x: record.PC1, y: record.PC2 })),
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }, {
                        label: 'Decision Boundary',
                        data: dividerData.x_values.map((x, i) => ({ x: x, y: dividerData.y_values[i] })),
                        borderColor: 'black',
                        borderWidth: 2,
                        type: 'line',
                        fill: false,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'PC1' } },
                        y: { title: { display: true, text: 'PC2' } }
                    }
                }
            });
        }

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>PCA Loan Data Visualization</h1>
    <p>This chart visualizes loan applicants' financial data using PCA. The blue points represent approved loans, while the red points indicate rejected loans. The black line represents the estimated decision boundary that separates the two groups.</p>
    <div class="chart-container">
        <canvas id="pcaChart"></canvas>
    </div>
</body>
</html>
"""

# Save HTML file
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"PCA visualization page saved to: {html_output_path}")
