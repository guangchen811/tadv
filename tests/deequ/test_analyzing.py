from cadv_exploration.utils import load_dotenv

load_dotenv()
import pandas as pd
from pydeequ.analyzers import *

from cadv_exploration.deequ import analyze_on_spark_df, spark_df_from_pandas_df
from cadv_exploration.loader import load_csv, load_csvs
from cadv_exploration.utils import get_project_root


def test_analyzing_on_small_dataset():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    spark_df, spark = spark_df_from_pandas_df(df)
    analyzers = [Size(), Completeness("a"), Completeness("b")]
    result_df = analyze_on_spark_df(spark, spark_df, analyzers=analyzers)
    assert result_df.shape[0] == 3
    assert result_df[result_df["name"] == "Size"]["value"].values[0] == 3
    assert result_df[result_df["name"] == "Completeness"]["value"].values[0] == 1.0
    assert result_df[result_df["name"] == "Completeness"]["value"].values[1] == 1.0


def test_analyzing_on_large_dataset():
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
    analyzers = [Size(), Completeness("Age"), Mean("Billing Amount")]
    result_df = analyze_on_spark_df(spark, spark_df, analyzers=analyzers)
    assert result_df.shape[0] == 3
    assert result_df[result_df["name"] == "Size"]["value"].values[0] == 55500
    assert result_df[result_df["name"] == "Completeness"]["value"].values[0] == 1.0
    assert round(result_df[result_df["name"] == "Mean"]["value"].values[0]) == 25539
