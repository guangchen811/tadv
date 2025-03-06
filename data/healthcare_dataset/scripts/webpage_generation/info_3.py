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
df["Medical Condition"] = df["Medical Condition"].astype(str)
df["Doctor"] = df["Doctor"].astype(str)
df["Full Name"] = df["Name"].astype(str)

# Convert date columns to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors='coerce')
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors='coerce')

# Calculate hospital stay duration
df["Days Stayed"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# Group data by Medical Condition
data_dict = {}
for condition, group in df.groupby("Medical Condition"):
    avg_age = group["Age"].mean()
    avg_days = group["Days Stayed"].mean()
    most_frequent_doctor = group["Doctor"].mode()[0] if not group["Doctor"].mode().empty else "Unknown"

    data_dict[condition] = {
        "Average Age": round(avg_age, 2),
        "Average Stay Days": round(avg_days, 2),
        "Most Frequent Doctor": most_frequent_doctor
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with a dropdown for medical conditions
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Medical Condition Statistics</title>
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
            resultDiv.innerHTML = "";

            if (data[condition]) {{
                let info = data[condition];
                resultDiv.innerHTML = `
                    <p>Average Age: ${{info["Average Age"]}}</p>
                    <p>Average Stay Days: ${{info["Average Stay Days"]}}</p>
                    <p>Most Frequent Doctor: ${{info["Most Frequent Doctor"]}}</p>
                `;
            }} else {{
                resultDiv.innerHTML = "No records found.";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Medical Condition Statistics</h1>

    <label for="conditionSelect">Select a Medical Condition:</label>
    <select id="conditionSelect" onchange="queryInfo()">
        <option value="" disabled selected>Choose a condition</option>
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
