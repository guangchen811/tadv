from cadv_exploration.utils import load_dotenv

load_dotenv()
from cadv_exploration.deequ_wrapper import DeequWrapper
from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.loader import FileLoader


def test_spark_df_to_column_desc(resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    deequ_wrapper = DeequWrapper()
    df = FileLoader.load_csv(file_path)
    spark_df, spark = deequ_wrapper.spark_df_from_pandas_df(df)
    yaml_string = spark_df_to_column_desc(spark_df, spark)
    assert yaml_string.startswith("UserName:\n  completeness: 1.0\n")
