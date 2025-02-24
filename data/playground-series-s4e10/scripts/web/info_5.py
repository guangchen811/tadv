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

# Generate HTML with interactive bar chart
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Loan Risk Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }
        .chart-container {
            width: 80%;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            background: #fff;
        }
    </style>
    <script>
        let data = [];

        function loadData() {
            fetch('data.json')
                .then(response => response.json())
                .then(jsonData => {
                    data = jsonData;
                    renderChart(jsonData);
                })
                .catch(error => console.error("Error loading data:", error));
        }

        function renderChart(data) {
            let ctx = document.getElementById("barChart").getContext('2d');
            let labels = data.map(row => row["Income Bracket"]);
            let defaultRates = data.map(row => row["Default Rate (%)"]);
            let totalLoans = data.map(row => row["Total_Loans"]);

            if (window.myChart) {
                window.myChart.destroy();
            }

            window.myChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Default Rate (%)',
                        data: defaultRates,
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(tooltipItem) {
                                    let index = tooltipItem.dataIndex;
                                    return `Default Rate: ${defaultRates[index].toFixed(2)}% \nTotal Loans: ${totalLoans[index]}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Default Rate (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Income Bracket'
                            }
                        }
                    }
                }
            });
        }

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Loan Risk Analysis: Default Rate by Income Bracket</h1>
    <div class="chart-container">
        <canvas id="barChart"></canvas>
    </div>
</body>
</html>
"""

# Save HTML file
html_output_path = os.path.join(args.output, "index.html")
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated Loan Risk Matrix JSON: {json_output_path}")
print(f"Generated Loan Risk Analysis Website with Interactive Bar Chart: {html_output_path}")
