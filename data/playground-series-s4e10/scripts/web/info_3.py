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


# Compute heatmap data
def compute_heatmap_data(x_col, y_col):
    heatmap_data = df.groupby([x_col, y_col]).size().unstack(fill_value=0)
    return heatmap_data


heatmaps = {
    "Loan Grade vs Loan Intent": compute_heatmap_data("loan_grade", "loan_intent"),
    "Loan Grade vs Loan Status": compute_heatmap_data("loan_grade", "loan_status"),
    "Home Ownership vs Loan Status": compute_heatmap_data("person_home_ownership", "loan_status"),
    "Employment Length vs Loan Status": compute_heatmap_data("person_emp_length", "loan_status")
}

# Remove empty heatmaps
heatmaps = {key: value for key, value in heatmaps.items() if not value.empty}

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
heatmap_json = {key: value.to_dict() for key, value in heatmaps.items()}
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(heatmap_json, f, indent=4)

# Generate HTML with Chart.js Matrix Plugin
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Loan Data Heatmap</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.1.0"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }}
        .chart-container {{
            width: 70%;
            height: 500px;
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
                    console.log("Loaded data:", jsonData);
                    data = jsonData;
                    populateDropdown();
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function populateDropdown() {{
            let dropdown = document.getElementById("heatmapSelect");
            dropdown.innerHTML = "<option value='' disabled selected>Choose a heatmap</option>";
            console.log("Data keys:", Object.keys(data));
            Object.keys(data).forEach(category => {{
                let option = document.createElement("option");
                option.value = category;
                option.textContent = category;
                dropdown.appendChild(option);
            }});
        }}

        function renderChart() {{
            let category = document.getElementById("heatmapSelect").value;
            if (!category || !data[category]) return;

            let ctx = document.getElementById("heatmapChart").getContext('2d');
            let rawData = data[category];
            let labelsX = Object.keys(rawData);
            let labelsY = [...new Set(Object.values(rawData).flatMap(row => Object.keys(row)))];

            let matrixData = [];
            labelsX.forEach((x, i) => {{
                labelsY.forEach((y, j) => {{
                    matrixData.push({{
                        x: i,
                        y: j,
                        v: rawData[x]?.[y] || 0
                    }});
                }});
            }});

            if (window.myChart) {{
                window.myChart.destroy();
            }}

            window.myChart = new Chart(ctx, {{
                type: 'matrix',
                data: {{
                    datasets: [{{
                        label: category,
                        data: matrixData,
                        backgroundColor: (ctx) => {{
                            let value = ctx.dataset.data[ctx.dataIndex].v;
                            return `rgba(255, 99, 132, ${{Math.min(value / 100, 1)}})`;
                        }},
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        width: (ctx) => ctx.chart.chartArea ? Math.min(ctx.chart.chartArea.width / labelsX.length, 40) : 20,
                        height: (ctx) => ctx.chart.chartArea ? Math.min(ctx.chart.chartArea.height / labelsY.length, 40) : 20
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            type: 'category',
                            labels: labelsX,
                            title: {{ display: true, text: 'X-Axis' }}
                        }},
                        y: {{
                            type: 'category',
                            labels: labelsY,
                            title: {{ display: true, text: 'Y-Axis' }}
                        }}
                    }}
                }}
            }});
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Loan Data Heatmap</h1>
    <label for="heatmapSelect">Select a Heatmap:</label>
    <select id="heatmapSelect" onchange="renderChart()">
    </select>
    <div class="chart-container">
        <canvas id="heatmapChart"></canvas>
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
