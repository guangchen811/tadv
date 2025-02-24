import argparse
import os

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # 1. Read CSV
    df = pd.read_csv(os.path.join(args.input, "new_data.csv"))

    # 2. Select features and target
    features = [
        "person_income",
        "loan_amnt",
        "person_emp_length",
        "cb_person_cred_hist_length",
        "loan_int_rate",
        "loan_percent_income"
    ]
    target = "loan_status"  # e.g., 1 = approved, 0 = not approved
    df_selected = df[features + [target]].dropna()

    # 3. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected[features])
    y = df_selected[target].values

    # 4. Apply TSNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)

    # 5. Build DataFrame
    df_tsne = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"])
    df_tsne["loan_status"] = y

    # 6. Save TSNE data as JSON
    tsne_json_path = os.path.join(args.output, "tsne_data.json")
    df_tsne.to_json(tsne_json_path, orient="records", indent=4)
    print(f"t-SNE data saved to: {tsne_json_path}")

    # 7. Create minimal HTML with Chart.js
    html_output_path = os.path.join(args.output, "tsne_visualization.html")
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>t-SNE Loan Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }}
        .chart-container {{
            width: 80%;
            max-width: 700px;
            margin: auto;
        }}
    </style>
    <script>
        let tsneData = [];

        function loadData() {{
            fetch('tsne_data.json')
                .then(response => response.json())
                .then(jsonData => {{
                    tsneData = jsonData;
                    renderChart();
                }})
                .catch(error => console.error("Error loading data:", error));
        }}

        function renderChart() {{
            let ctx = document.getElementById("tsneChart").getContext('2d');

            if (window.myChart) {{
                window.myChart.destroy();
            }}

            let approved = tsneData.filter(d => d.loan_status === 1);
            let rejected = tsneData.filter(d => d.loan_status === 0);

            window.myChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{
                    datasets: [
                        {{
                            label: 'Approved Loans',
                            data: approved.map(d => {{
                                return {{x: d.TSNE1, y: d.TSNE2}};
                            }}),
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Rejected Loans',
                            data: rejected.map(d => {{
                                return {{x: d.TSNE1, y: d.TSNE2}};
                            }}),
                            backgroundColor: 'rgba(255, 99, 132, 0.6)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'TSNE1' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'TSNE2' }}
                        }}
                    }}
                }}
            }});
        }}

        window.onload = loadData;
    </script>
</head>
<body>
    <h1>t-SNE Loan Data Visualization</h1>
    <p>
       A simple 2D t-SNE scatter plot. 
       Blue = Approved Loans, 
       Red = Rejected Loans.
    </p>
    <div class="chart-container">
        <canvas id="tsneChart"></canvas>
    </div>
</body>
</html>
"""
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"t-SNE visualization page saved to: {html_output_path}")


if __name__ == "__main__":
    main()
