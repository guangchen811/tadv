from tadv.utils import load_dotenv

load_dotenv()
import pandas as pd

from tadv.loader import FileLoader


def test_profiling_on_small_dataset(dq_manager):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    result_df = dq_manager.profile_on_spark_df(spark, spark_df).profiles
    result_df_a = result_df["a"]
    assert result_df_a.completeness == 1.0
    assert result_df_a.dataType == "Integral"


def test_spark_df_from_local_csv(dq_manager, resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    result_df = dq_manager.profile_on_spark_df(spark, spark_df).profiles
    result_df_billing_amount = result_df["FirstName"]
    assert result_df_billing_amount.completeness == 1.0
    assert result_df_billing_amount.approximateNumDistinctValues == 5
    assert result_df_billing_amount.dataType == "String"
