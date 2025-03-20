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

df = df.dropna(subset=["Name"])

# Ensure Full Name is treated as a string
df["Full Name"] = df["Name"].astype(str)

# Convert to dictionary format for efficient lookup
data_dict = {row["Full Name"].lower(): row.to_dict() for _, row in df.iterrows()}

# Save as JSON
json_output_path = os.path.join(args.output, "data.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4)

# Generate HTML without embedding full data
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Patient Information Query</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            display: none;
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
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function queryTable() {{
            let input = document.getElementById("nameQuery").value.toLowerCase().trim();
            let tableBody = document.getElementById("tableBody");
            let table = document.getElementById("dataTable");
            tableBody.innerHTML = "";

            if (input in data) {{
                let rowData = data[input];
                let rowHTML = "<tr>";
                Object.keys(rowData).forEach(key => {{
                    rowHTML += `<td>${{rowData[key]}}</td>`;
                }});
                rowHTML += "</tr>";
                tableBody.innerHTML = rowHTML;
                table.style.display = "table";
            }} else {{
                table.style.display = "none";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Patient Information Query</h1>

    <label for="nameQuery">Enter Full Name:</label>
    <input type="text" id="nameQuery" onkeyup="queryTable()" placeholder="Type to search...">

    <h2>Patient Records</h2>
    <table id="dataTable" class="table table-striped" border="0">
        <thead>
            <tr>
                {"".join(f'<th>{col}</th>' for col in df.columns)}
            </tr>
        </thead>
        <tbody id="tableBody">
        </tbody>
    </table>
</body>
</html>
"""

# Write HTML file
html_output_path = os.path.join(args.output, 'output.html')
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated static site: {html_output_path}")
print(f"Generated JSON data file: {json_output_path}")
