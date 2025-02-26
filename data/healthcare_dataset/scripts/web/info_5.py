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
df["Blood Type"] = df["Blood Type"].astype(str).str.strip().str.upper()
df["Gender"] = df["Gender"].astype(str).str.strip()
df["Medication"] = df["Medication"].astype(str).str.strip()
df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors='coerce')

# Standardize Blood Type formatting
blood_type_mapping = {
    "O PLUS": "O+", "O PLUS ": "O+", "O POS": "O+", "O POSITIVE": "O+", "O-PLUS": "O+", "O PLUS": "O+",
    "O MINUS": "O-", "O NEG": "O-", "O NEGATIVE": "O-", "O-MINUS": "O-",
    "A PLUS": "A+", "A POS": "A+", "A POSITIVE": "A+", "A-PLUS": "A+",
    "A MINUS": "A-", "A NEG": "A-", "A NEGATIVE": "A-", "A-MINUS": "A-",
    "B PLUS": "B+", "B POS": "B+", "B POSITIVE": "B+", "B-PLUS": "B+",
    "B MINUS": "B-", "B NEG": "B-", "B NEGATIVE": "B-", "B-MINUS": "B-",
    "AB PLUS": "AB+", "AB POS": "AB+", "AB POSITIVE": "AB+", "AB-PLUS": "AB+",
    "AB MINUS": "AB-", "AB NEG": "AB-", "AB NEGATIVE": "AB-", "AB-MINUS": "AB-"
}
df["Blood Type"] = df["Blood Type"].replace(blood_type_mapping)

# Group data by Blood Type
data_dict = {}
for blood_type, group in df.groupby("Blood Type"):
    gender_distribution = group["Gender"].value_counts().to_dict()
    avg_billing = group["Billing Amount"].mean()
    most_frequent_medication = group["Medication"].mode()[0] if not group["Medication"].mode().empty else "Unknown"

    data_dict[blood_type] = {
        "Gender Distribution": gender_distribution,
        "Average Billing Amount": round(avg_billing, 2),
        "Most Frequent Medication": most_frequent_medication
    }

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML with a dropdown for blood types
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Blood Type Statistics</title>
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
            let dropdown = document.getElementById("bloodTypeSelect");
            dropdown.innerHTML = "";
            Object.keys(data).forEach(bloodType => {{
                let option = document.createElement("option");
                option.value = bloodType;
                option.textContent = bloodType;
                dropdown.appendChild(option);
            }});
        }}

        function queryInfo() {{
            let bloodType = document.getElementById("bloodTypeSelect").value;
            let resultDiv = document.getElementById("result");
            resultDiv.innerHTML = "";

            if (data[bloodType]) {{
                let info = data[bloodType];
                let genderDist = Object.entries(info["Gender Distribution"]).map(([gender, count]) => `${{gender}}: ${{count}}`).join(', ');
                resultDiv.innerHTML = `
                    <p>Gender Distribution: ${{genderDist}}</p>
                    <p>Average Billing Amount: $${{info["Average Billing Amount"]}}</p>
                    <p>Most Frequent Medication: ${{info["Most Frequent Medication"]}}</p>
                `;
            }} else {{
                resultDiv.innerHTML = "No records found.";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Blood Type Statistics</h1>

    <label for="bloodTypeSelect">Select a Blood Type:</label>
    <select id="bloodTypeSelect" onchange="queryInfo()">
        <option value="" disabled selected>Choose a blood type</option>
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
