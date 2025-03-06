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
df["Department"] = df["Hospital"].astype(str)
df["Admission Type"] = df["Admission Type"].astype(str)
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors='coerce')
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

# Group data by Department
data_dict = {}
for department, group in df.groupby("Department"):
    avg_age = group["Age"].mean()
    most_common_admission = group["Admission Type"].mode()[0] if not group["Admission Type"].mode().empty else "Unknown"
    avg_billing = group["Billing Amount"].mean()

    data_dict[department] = {
        "Average Age": round(avg_age, 2),
        "Most Common Admission Type": most_common_admission,
        "Average Billing Amount": round(avg_billing, 2)
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with a dropdown for departments
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Hospital Department Statistics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .result {{
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
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
            let dropdown = document.getElementById("departmentSelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(department => {{
                let option = document.createElement("option");
                option.value = department;
                option.textContent = department;
                dropdown.appendChild(option);
            }});
        }}

        function queryInfo() {{
            let department = document.getElementById("departmentSelect").value;
            let resultDiv = document.getElementById("result");
            resultDiv.innerHTML = "";

            if (data[department]) {{
                let info = data[department];
                resultDiv.innerHTML = `
                    <p>Average Age: ${{info["Average Age"]}}</p>
                    <p>Most Common Admission Type: ${{info["Most Common Admission Type"]}}</p>
                    <p>Average Billing Amount: $${{info["Average Billing Amount"]}}</p>
                `;
            }} else {{
                resultDiv.innerHTML = "No records found.";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Hospital Department Statistics</h1>

    <label for="departmentSelect">Select a Department:</label>
    <select id="departmentSelect" onchange="queryInfo()">
        <option value="" disabled selected>Choose a department</option>
    </select>

    <div id="result" class="result"></div>
</body>
</html>
"""

# Write HTML file
html_output_path = os.path.join(args.output, 'output.html')
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated static site: {html_output_path}")
print(f"Generated JSON data file: {json_output_path}")
