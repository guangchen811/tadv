from cadv_exploration.utils import load_dotenv

load_dotenv()
import pandas as pd
from pydeequ.analyzers import *

from cadv_exploration.loader import FileLoader


def test_analyzing_on_small_dataset(dq_manager):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    analyzers = [Size(), Completeness("a"), Completeness("b")]
    result_df = dq_manager.analyze_on_spark_df(spark, spark_df, analyzers=analyzers)
    assert result_df.shape[0] == 3
    assert result_df[result_df["name"] == "Size"]["value"].values[0] == 3
    assert result_df[result_df["name"] == "Completeness"]["value"].values[0] == 1.0
    assert result_df[result_df["name"] == "Completeness"]["value"].values[1] == 1.0


def test_analyzing_on_local_dataset(dq_manager, resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    analyzers = [Size(), Completeness("Identifier"), Completeness("UserName")]
    result_df = dq_manager.analyze_on_spark_df(spark, spark_df, analyzers=analyzers)
    assert result_df.shape[0] == 3
    assert result_df[result_df["name"] == "Size"]["value"].values[0] == 5
    assert result_df[result_df["name"] == "Completeness"]["value"].values[0] == 1.0
    assert round(result_df[result_df["name"] == "Completeness"]["value"].values[0]) == 1.0
