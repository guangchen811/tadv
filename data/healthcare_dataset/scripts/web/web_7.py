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
df["Doctor"] = df["Doctor"].astype(str).str.strip()
df["Medical Condition"] = df["Medical Condition"].astype(str).str.strip()
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors='coerce')
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors='coerce')

# Calculate hospital stay duration
df["Days Stayed"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# Group data by Doctor
data_dict = {}
for doctor, group in df.groupby("Doctor"):
    total_patients = len(group)
    avg_stay_days = group["Days Stayed"].mean()
    most_common_condition = group["Medical Condition"].mode()[0] if not group[
        "Medical Condition"].mode().empty else "Unknown"

    data_dict[doctor] = {
        "Total Patients Treated": total_patients,
        "Average Stay Duration": round(avg_stay_days, 2),
        "Most Common Medical Condition": most_common_condition
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with a dropdown for doctors
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Doctor Performance Dashboard</title>
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
            let dropdown = document.getElementById("doctorSelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(doctor => {{
                let option = document.createElement("option");
                option.value = doctor;
                option.textContent = doctor;
                dropdown.appendChild(option);
            }});
        }}

        function queryInfo() {{
            let doctor = document.getElementById("doctorSelect").value;
            let resultDiv = document.getElementById("result");
            resultDiv.innerHTML = "";

            if (data[doctor]) {{
                let info = data[doctor];
                resultDiv.innerHTML = `
                    <p>Total Patients Treated: ${{info["Total Patients Treated"]}}</p>
                    <p>Average Stay Duration: ${{info["Average Stay Duration"]}} days</p>
                    <p>Most Common Medical Condition: ${{info["Most Common Medical Condition"]}}</p>
                `;
            }} else {{
                resultDiv.innerHTML = "No records found.";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Doctor Performance Dashboard</h1>

    <label for="doctorSelect">Select a Doctor:</label>
    <select id="doctorSelect" onchange="queryInfo()">
        <option value="" disabled selected>Choose a doctor</option>
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
