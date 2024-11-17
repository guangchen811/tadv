from cadv_exploration.utils import load_dotenv

load_dotenv()

import pandas as pd

from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


def test_spark_df_from_pandas_df(deequ_wrapper):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
    spark_df, _ = deequ_wrapper.spark_df_from_pandas_df(df)
    assert spark_df.count() == 3
    assert spark_df.columns == ["a", "b"]
    assert spark_df.collect()[0].a == 1
    assert spark_df.collect()[0].b == "foo"


def test_spark_df_from_local_csv(deequ_wrapper):
    project_root = get_project_root()
    file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "files"
            / "healthcare_dataset.csv"
    )
    df = FileLoader.load_csv(file_path)
    spark_df, _ = deequ_wrapper.spark_df_from_pandas_df(df)
    assert spark_df.count() == 55500
