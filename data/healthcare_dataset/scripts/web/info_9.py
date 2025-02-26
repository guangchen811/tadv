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

# Ensure necessary columns are treated as strings
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors='coerce')
df["Admission Type"] = df["Admission Type"].astype(str).str.strip()

# Extract year and month
df["Year"] = df["Date of Admission"].dt.year
df["Month"] = df["Date of Admission"].dt.month

# Group data by year
data_dict = {}
for year, group in df.groupby("Year"):
    total_admissions = len(group)
    monthly_distribution = group["Month"].value_counts().sort_index().to_dict()
    admission_types = group["Admission Type"].value_counts().to_dict()

    data_dict[year] = {
        "Total Admissions": total_admissions,
        "Monthly Distribution": monthly_distribution,
        "Admission Types": admission_types
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with a dropdown for year selection and a chart display
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Patient Admission Trends</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .chart-container {{
            margin-top: 20px;
            width: 80%;
        }}
    </style>
    <script>
        let data = {{}};

        function loadData() {{
            fetch('data.json')
                .then(response => response.json())
                .then(jsonData => {{
                    data = jsonData;
                    console.log("Data loaded successfully.");
                    populateDropdown();
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function populateDropdown() {{
            let dropdown = document.getElementById("yearSelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(year => {{
                let option = document.createElement("option");
                option.value = year;
                option.textContent = year;
                dropdown.appendChild(option);
            }});
        }}

        function queryInfo() {{
            let year = document.getElementById("yearSelect").value;
            let resultDiv = document.getElementById("result");
            let chartContainer = document.getElementById("chartContainer");
            let ctx = document.getElementById("admissionChart").getContext('2d');
            resultDiv.innerHTML = "";

            if (data[year]) {{
                let info = data[year];
                resultDiv.innerHTML = `<p>Total Admissions: ${{info["Total Admissions"]}}</p>`;

                let labels = Object.keys(info["Monthly Distribution"]).map(m => "Month " + m);
                let values = Object.values(info["Monthly Distribution"]);

                chartContainer.style.display = "block";

                if (window.myChart) {{
                    window.myChart.destroy();
                }}

                window.myChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{label: 'Monthly Admissions',
                            data: values,
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false
                    }}
                }});
            }} else {{
                chartContainer.style.display = "none";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Patient Admission Trends</h1>

    <label for="yearSelect">Select a Year:</label>
    <select id="yearSelect" onchange="queryInfo()">
        <option value="" disabled selected>Choose a year</option>
    </select>

    <div id="result" class="result"></div>

    <div id="chartContainer" class="chart-container" style="display: none;">
        <canvas id="admissionChart"></canvas>
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
