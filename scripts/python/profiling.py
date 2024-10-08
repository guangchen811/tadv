import os
import pandas as pd
from cadv_exploration.utils import get_project_root
from cadv_exploration.loader import load_csv

os.environ["SPARK_VERSION"] = "3.5"
from cadv_exploration.deequ import spark_df_from_pandas_df, profile_on_spark_df

from pydeequ.analyzers import *


def main():
    project_root = get_project_root()
    file_path = (
        project_root
        / "data"
        / "prasad22"
        / "healthcare-dataset"
        / "files"
        / "healthcare_dataset.csv"
    )
    df = load_csv(file_path)
    spark_df, spark = spark_df_from_pandas_df(df)
    result_df = profile_on_spark_df(spark, spark_df).profiles
    for col, profile in result_df.items():
        print(type(profile))


if __name__ == "__main__":
    main()
