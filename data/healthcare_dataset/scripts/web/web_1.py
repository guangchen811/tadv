import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# Read CSV into a DataFrame
df = pd.read_csv(args.input + "/train.csv")

# Filter to rows where person_age > 20
filtered_df = df[df["Age"] > 20]

# Take only the first 10 rows (or fewer if not enough rows)
limited_df = filtered_df.head(10)

# Convert the top-10 filtered DataFrame to an HTML table
table_html = limited_df.to_html(index=False, classes="table table-striped", border=0)

# Generate descriptive stats for the filtered subset (not just the top 10, but the entire filtered set)
# If you want stats only on the 10 rows, use limited_df instead
stats_df = filtered_df.describe(include="all")
stats_html = stats_df.to_html(classes="table table-striped", border=0)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Filtered Loan Data Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        .table th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>Loan Data Dashboard (Filtered by Age > 20)</h1>

    <p><strong>Total Rows with Age > 20:</strong> {len(filtered_df)}</p>

    <h2>Top 10 Matching Records</h2>
    {table_html}

    <h2>Descriptive Statistics for All Filtered Rows</h2>
    {stats_html}
</body>
</html>
"""
# Write to an HTML file
with open(args.output + "/" + 'output.html', "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Generated static site: {args.output}/output.html")
