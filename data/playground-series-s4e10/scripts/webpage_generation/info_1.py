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

# Extract categorical column frequencies
data_dict = {}
categorical_columns = ["loan_grade", "loan_intent", "cb_person_default_on_file"]

for column in categorical_columns:
    data_dict[column] = df[column].value_counts().to_dict()

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
    <title>Loan Data Explorer</title>
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
        select {{
            padding: 10px;
            font-size: 16px;
            margin-top: 10px;
            border-radius: 5px;
        }}
    </style>
    <script>
        let data = {{}};

        function loadData() {{
            fetch('data.json')
                .then(response => response.json())
                .then(jsonData => {{
                    data = jsonData;
                    populateDropdown();
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function populateDropdown() {{
            let dropdown = document.getElementById("categorySelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(category => {{
                let option = document.createElement("option");
                option.value = category;
                option.textContent = category.replace('_', ' ').toUpperCase();
                dropdown.appendChild(option);
            }});
        }}

        function renderChart() {{
            let category = document.getElementById("categorySelect").value;
            let ctx = document.getElementById("barChart").getContext('2d');
            let labels = Object.keys(data[category]);
            let values = Object.values(data[category]);

            if (window.myChart) {{
                window.myChart.destroy();
            }}

            window.myChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
label: `Frequency of ${{category.replace('_', ' ').toUpperCase()}}`,
                        data: values,
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false
                }}
            }});
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Loan Data Explorer</h1>
    <label for="categorySelect">Select a Category:</label>
    <select id="categorySelect" onchange="renderChart()">
        <option value="" disabled selected>Choose a category</option>
    </select>
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
