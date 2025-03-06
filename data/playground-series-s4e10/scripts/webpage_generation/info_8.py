import argparse
import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# Read CSV into DataFrame
df = pd.read_csv(os.path.join(args.input, "new_data.csv"))

# Select numerical features for PCA analysis
features = ["person_income", "loan_amnt", "person_emp_length", "cb_person_cred_hist_length", "loan_int_rate",
            "loan_percent_income"]
df_selected = df[features].dropna()

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
df_pca["record_id"] = df.index[:len(df_pca)]  # Assign record IDs

# Select 10 records for highlighting in the dropdown
df_highlighted = df_pca.head(10)
df_selected_highlighted = df_selected.iloc[df_highlighted.index]

# Merge PCA results with original features for better explanations
df_pca_full = df_pca.join(df_selected.reset_index(drop=True))
df_highlighted_full = df_highlighted.join(df_selected_highlighted.reset_index(drop=True))

# Save PCA-transformed data as JSON
pca_json_path = os.path.join(args.output, "pca_data.json")
df_pca_full.to_json(pca_json_path, orient="records", indent=4)

highlight_json_path = os.path.join(args.output, "highlighted_pca_data.json")
df_highlighted_full.to_json(highlight_json_path, orient="records", indent=4)

print(f"PCA data saved to: {pca_json_path}")
print(f"Highlighted PCA data saved to: {highlight_json_path}")

# Generate HTML for PCA visualization
html_output_path = os.path.join(args.output, "index.html")
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>PCA Loan Data Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }
        .container {
            width: 80%;
            margin: auto;
        }
        select {
            padding: 10px;
            font-size: 16px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .chart-container {
            width: 100%;
            max-width: 600px;
            margin: auto;
        }
        table {
            width: 80%;
            margin: 20px auto;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #f4f4f4;
        }
    </style>
    <script>
        let pcaData = [];
        let highlightedData = [];

        function loadData() {
            fetch('pca_data.json')
                .then(response => response.json())
                .then(jsonData => {
                    pcaData = jsonData;
                    fetch('highlighted_pca_data.json')
                        .then(response => response.json())
                        .then(highlightedJson => {
                            highlightedData = highlightedJson;
                            populateDropdown();
                        });
                })
                .catch(error => console.error("Error loading data:", error));
        }

        function populateDropdown() {
            let dropdown = document.getElementById("recordSelect");
            dropdown.innerHTML = "";
            highlightedData.forEach((record, index) => {
                let option = document.createElement("option");
                option.value = index;
                option.textContent = "Record " + record.record_id;
                dropdown.appendChild(option);
            });
        }

        function renderChart() {
            let selectedIndex = document.getElementById("recordSelect").value;
            let selectedRecord = highlightedData[selectedIndex];

            let ctx = document.getElementById("pcaChart").getContext('2d');
            if (window.myChart) {
                window.myChart.destroy();
            }
            window.myChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Selected Record',
                        data: [{ x: selectedRecord.PC1, y: selectedRecord.PC2 }],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 3,
                        pointRadius: 6
                    }, {
                        label: 'PCA Projection',
                        data: pcaData.map(record => ({ x: record.PC1, y: record.PC2 })),
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'PC1' } },
                        y: { title: { display: true, text: 'PC2' } }
                    }
                }
            });
            renderTable(selectedRecord);
        }

        function renderTable(record) {
            let tableBody = document.getElementById("recordDetails");
            tableBody.innerHTML = "";
            Object.keys(record).forEach(key => {
                if (key !== "PC1" && key !== "PC2" && key !== "record_id") {
                    let row = `<tr><th>${key}</th><td>${record[key]}</td></tr>`;
                    tableBody.innerHTML += row;
                }
            });
        }

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>PCA Loan Data Visualization</h1>
    <p>This tool allows you to explore loan data using Principal Component Analysis (PCA). The scatter plot represents the reduced two-dimensional projection of multiple financial features. You can select one of the highlighted records from the dropdown to view its position and details.</p>
    <select id="recordSelect" onchange="renderChart()">
        <option value="" disabled selected>Choose a record</option>
    </select>
    <div class="chart-container">
        <canvas id="pcaChart"></canvas>
    </div>
    <h2>Record Details</h2>
    <table>
        <tbody id="recordDetails"></tbody>
    </table>
</body>
</html>
"""

# Save HTML file
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"PCA visualization page saved to: {html_output_path}")
