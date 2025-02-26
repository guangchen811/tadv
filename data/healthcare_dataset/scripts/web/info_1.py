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

# Ensure Full Name is treated as a string
df["Full Name"] = df["Name"].astype(str)

# Convert date columns to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors='coerce')
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors='coerce')

# Calculate hospital stay duration
df["Days Stayed"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# Convert to dictionary format for efficient lookup
data_dict = {row["Full Name"].lower(): {"Age": row["Age"], "Days Stayed": row["Days Stayed"]} for _, row in df.iterrows()}

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
    <title>Hospital Stay Query</title>
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
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function queryInfo() {{
            let input = document.getElementById("nameQuery").value.toLowerCase().trim();
            let resultDiv = document.getElementById("result");
            resultDiv.innerHTML = "";

            if (data[input]) {{
                let info = data[input];
                resultDiv.innerHTML = `Age: ${{info["Age"]}} <br> Days Stayed: ${{info["Days Stayed"]}}`;
            }} else {{
                resultDiv.innerHTML = "No records found.";
            }}
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>Hospital Stay Query</h1>

    <label for="nameQuery">Enter Full Name:</label>
    <input type="text" id="nameQuery" onkeyup="queryInfo()" placeholder="Type to search...">

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