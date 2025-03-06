import argparse
import json
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# Read CSV into a DataFrame
df = pd.read_csv(os.path.join(args.input, "new_data.csv"))

# Compute loan approval rate by loan grade
approval_rates = df.groupby(["loan_grade", "loan_status"]).size().unstack(fill_value=0)
approval_rates["approval_rate"] = approval_rates[1] / (approval_rates[0] + approval_rates[1])
data_dict = approval_rates["approval_rate"].to_dict()

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with interactive dropdowns and bar charts
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Loan Approval Rate by Loan Grade</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }}
        .chart-container {{
            width: 60%;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }}
    </style>
    <script>
        let data = {{}};

        function loadData() {{
            fetch('data.json')
                .then(response => response.json())
                .then(jsonData => {{
                    data = jsonData;
                    renderChart();
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function renderChart() {{
            let ctx = document.getElementById("barChart").getContext('2d');
            let labels = Object.keys(data);
            let values = Object.values(data);

            if (window.myChart) {{
                window.myChart.destroy();
            }}

            window.myChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Loan Approval Rate',
                        data: values,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1,
                            title: {{
                                display: true,
                                text: 'Approval Rate'
                            }}
                        }}
                    }}
                }}
            }});
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Loan Approval Rate by Loan Grade</h1>
    <div class="chart-container">
        <canvas id="barChart"></canvas>
    </div>
</body>
</html>
"""

# Write HTML file
html_output_path = os.path.join(args.output, 'output.html')
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated static site: {html_output_path}")
print(f"Generated JSON data file: {json_output_path}")
