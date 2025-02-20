import argparse
import pandas as pd
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# Read CSV into a DataFrame
df = pd.read_csv(os.path.join(args.input, "new_data.csv"))

# Ensure necessary columns are treated as strings
df["Medical Condition"] = df["Medical Condition"].astype(str).str.strip()
df["Medication"] = df["Medication"].astype(str).str.strip()
df["Hospital"] = df["Hospital"].astype(str).str.strip()
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

# Group data by Medical Condition
data_dict = {}
for condition, group in df.groupby("Medical Condition"):
    most_common_medication = group["Medication"].mode()[0] if not group["Medication"].mode().empty else "Unknown"
    most_common_hospital = group["Hospital"].mode()[0] if not group["Hospital"].mode().empty else "Unknown"
    age_distribution = group["Age"].value_counts().sort_index().to_dict()

    data_dict[condition] = {
        "Most Common Medication": most_common_medication,
        "Most Common Hospital": most_common_hospital,
        "Age Distribution": age_distribution
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with enhanced UI and interactive chart
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Disease & Medication Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
            text-align: center;
        }}
        .container {{
            width: 80%;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }}
        .chart-container {{
            margin-top: 20px;
            width: 100%;
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
                    console.log("Data loaded successfully.");
                    populateDropdown();
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function populateDropdown() {{
            let dropdown = document.getElementById("conditionSelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(condition => {{
                let option = document.createElement("option");
                option.value = condition;
                option.textContent = condition;
                dropdown.appendChild(option);
            }});
        }}

        function queryInfo() {{
            let condition = document.getElementById("conditionSelect").value;
            let resultDiv = document.getElementById("result");
            let chartContainer = document.getElementById("chartContainer");
            let ctx = document.getElementById("ageChart").getContext('2d');
            resultDiv.innerHTML = "";

            if (data[condition]) {{
                let info = data[condition];
                resultDiv.innerHTML = `<p><strong>Most Common Medication:</strong> ${{info["Most Common Medication"]}}</p>
                                      <p><strong>Most Common Hospital:</strong> ${{info["Most Common Hospital"]}}</p>`;

                let labels = Object.keys(info["Age Distribution"]);
                let values = Object.values(info["Age Distribution"]);

                chartContainer.style.display = "block";

                if (window.myChart) {{
                    window.myChart.destroy();
                }}

                window.myChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: labels,
                        datasets: [{{
label: 'Age Distribution',
                            data: values,
                            backgroundColor: 'rgba(255, 99, 132, 0.6)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            fill: true
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
    <div class="container">
        <h1>Disease & Medication Explorer</h1>

        <label for="conditionSelect">Select a Medical Condition:</label>
        <select id="conditionSelect" onchange="queryInfo()">
            <option value="" disabled selected>Choose a condition</option>
        </select>

        <div id="result" class="result"></div>

        <div id="chartContainer" class="chart-container" style="display: none;">
            <canvas id="ageChart"></canvas>
        </div>
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