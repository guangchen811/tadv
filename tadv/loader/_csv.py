import os
from typing import List

import pandas as pd


def load_csvs(dir_path: str) -> List[pd.DataFrame]:
    """Load a list of CSV files into a list of pandas DataFrames."""
    file_path = [f"{dir_path}/{file}" for file in os.listdir(dir_path)]
    return [load_csv(file) for file in file_path]


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, **kwargs)
