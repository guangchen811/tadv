import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# Read CSV into a DataFrame
df = pd.read_csv(os.path.join(args.input, "new_data.csv"))

# Define income brackets
income_bins = [0, 20000, 50000, 100000, 200000, float("inf")]
income_labels = ["< $20k", "$20k-$50k", "$50k-$100k", "$100k-$200k", "> $200k"]
df["Income Bracket"] = pd.cut(df["person_income"], bins=income_bins, labels=income_labels, right=False)

# Compute loan risk matrix
loan_risk_matrix = df.groupby("Income Bracket").agg(
    Total_Loans=("loan_amnt", "count"),
    Avg_Loan_Amount=("loan_amnt", "mean"),
    Defaults=("loan_status", lambda x: (x == 1).sum()),
)

# Compute default rates
loan_risk_matrix["Default Rate (%)"] = (loan_risk_matrix["Defaults"] / loan_risk_matrix["Total_Loans"]) * 100
loan_risk_matrix = loan_risk_matrix.reset_index()

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
loan_risk_matrix.to_json(json_output_path, orient="records", indent=4)

# Generate HTML with interactive table
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Loan Risk Matrix</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }
        .table-container {
            width: 80%;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            background: #fff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: center;
        }
        th {
            background-color: #f4f4f4;
        }
    </style>
    <script>
        function loadData() {
            fetch('data.json')
                .then(response => response.json())
                .then(jsonData => {
                    populateTable(jsonData);
                })
                .catch(error => console.error("Error loading data:", error));
        }

        function populateTable(data) {
            let tableBody = document.getElementById("tableBody");
            tableBody.innerHTML = "";

            data.forEach(row => {
                let tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${row["Income Bracket"]}</td>
                    <td>${row["Total_Loans"]}</td>
                    <td>${row["Avg_Loan_Amount"].toFixed(2)}</td>
                    <td>${row["Defaults"]}</td>
                    <td>${row["Default Rate (%)"].toFixed(2)}%</td>
                `;
                tableBody.appendChild(tr);
            });
        }

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Loan Risk Matrix: Income, Loan Amount & Default Rate</h1>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Income Bracket</th>
                    <th>Total Loans</th>
                    <th>Avg Loan Amount ($)</th>
                    <th>Defaults</th>
                    <th>Default Rate (%)</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>
</body>
</html>
"""

# Save HTML file
html_output_path = os.path.join(args.output, "index.html")
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated Loan Risk Matrix JSON: {json_output_path}")
print(f"Generated Loan Risk Matrix Website: {html_output_path}")
