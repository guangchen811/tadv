import os
import pandas as pd
from cadv_exploration.utils import get_project_root
from cadv_exploration.loader import load_csv

os.environ["SPARK_VERSION"] = "3.5"
from cadv_exploration.deequ import spark_df_from_pandas_df, profile_on_spark_df

from pydeequ.analyzers import *


def test_profiling_on_small_dataset():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    spark_df, spark = spark_df_from_pandas_df(df)
    result_df = profile_on_spark_df(spark, spark_df).profiles
    result_df_a = result_df["a"]
    assert result_df_a.completeness == 1.0
    assert result_df_a.dataType == "Integral"


def test_spark_df_from_local_csv():
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
    result_df_billing_amount = result_df["Billing Amount"]
    assert result_df_billing_amount.completeness == 1.0
    assert result_df_billing_amount.approximateNumDistinctValues == 49028
    assert result_df_billing_amount.dataType == 'Fractional'
