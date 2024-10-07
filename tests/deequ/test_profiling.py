import os
import pandas as pd

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
