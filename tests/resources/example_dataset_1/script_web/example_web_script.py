import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()
# %%
# Load the data
df = pd.read_csv(args.input + "/example_table.csv")

# Perform some operations
df["FullName"] = df["FirstName"] + " " + df["LastName"]

# Save the data
df.to_csv(args.output + "/output.csv", index=False)
