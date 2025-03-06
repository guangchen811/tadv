import argparse
import os

import pandas as pd
from sklearn.decomposition import PCA
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

# Save PCA-transformed data with loan approval status as JSON
pca_json_path = os.path.join(args.output, "pca_all_data.json")
df_pca.to_json(pca_json_path, orient="records", indent=4)

print(f"PCA data saved to: {pca_json_path}")

# Generate HTML for PCA visualization with color-coded approval status
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

        function loadData() {
            fetch('pca_all_data.json')
                .then(response => response.json())
                .then(jsonData => {
                    pcaData = jsonData;
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
    <p>This chart visualizes loan applicants' financial data using PCA. The blue points represent approved loans, while the red points indicate rejected loans.</p>
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
