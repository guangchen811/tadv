from tadv.utils import load_dotenv

load_dotenv()

import pandas as pd

from tadv.loader import FileLoader


def test_spark_df_from_pandas_df(dq_manager):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    spark_df, _ = dq_manager.spark_df_from_pandas_df(df)
    assert spark_df.count() == 3
    assert spark_df.columns == ["a", "b"]
    assert spark_df.collect()[0].a == 1
    assert spark_df.collect()[0].b == "foo"


def test_spark_df_from_local_csv(dq_manager, resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, _ = dq_manager.spark_df_from_pandas_df(df)
    assert spark_df.count() == 5
