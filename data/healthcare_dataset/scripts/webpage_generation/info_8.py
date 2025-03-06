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
df["Insurance Provider"] = df["Insurance Provider"].astype(str).str.strip()
df["Hospital"] = df["Hospital"].astype(str).str.strip()
df["Medical Condition"] = df["Medical Condition"].astype(str).str.strip()
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors='coerce')

# Group data by Insurance Provider
data_dict = {}
for provider, group in df.groupby("Insurance Provider"):
    avg_billing = group["Billing Amount"].mean()
    most_common_treatment = group["Medical Condition"].mode()[0] if not group[
        "Medical Condition"].mode().empty else "Unknown"
    most_common_hospital = group["Hospital"].mode()[0] if not group["Hospital"].mode().empty else "Unknown"

    data_dict[provider] = {
        "Average Billing Amount": round(avg_billing, 2),
        "Most Common Treatment": most_common_treatment,
        "Most Frequent Hospital": most_common_hospital
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with a dropdown for insurance providers and table format output
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Insurance Coverage Insights</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .table-container {{
            margin-top: 20px;
            display: none;
        }}
        .table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        .table th {{
            background-color: #f2f2f2;
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
            let dropdown = document.getElementById("insuranceSelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(provider => {{
                let option = document.createElement("option");
                option.value = provider;
                option.textContent = provider;
                dropdown.appendChild(option);
            }});
        }}

        function queryInfo() {{
            let provider = document.getElementById("insuranceSelect").value;
            let tableContainer = document.getElementById("tableContainer");
            let tableBody = document.getElementById("tableBody");
            tableBody.innerHTML = "";

            if (data[provider]) {{
                let info = data[provider];
                let rowHTML = `<tr>
                    <td>${{info["Average Billing Amount"]}}</td>
                    <td>${{info["Most Common Treatment"]}}</td>
                    <td>${{info["Most Frequent Hospital"]}}</td>
                </tr>`;
                tableBody.innerHTML = rowHTML;
                tableContainer.style.display = "block";
            }} else {{
                tableContainer.style.display = "none";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Insurance Coverage Insights</h1>

    <label for="insuranceSelect">Select an Insurance Provider:</label>
    <select id="insuranceSelect" onchange="queryInfo()">
        <option value="" disabled selected>Choose a provider</option>
    </select>

    <div id="tableContainer" class="table-container">
        <h2>Insurance Statistics</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Average Billing Amount</th>
                    <th>Most Common Treatment</th>
                    <th>Most Frequent Hospital</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
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
