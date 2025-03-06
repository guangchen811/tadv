from tadv.utils import load_dotenv

load_dotenv()
from tadv.dq_manager import DeequDataQualityManager
from tadv.inspector.deequ.deequ_inspector_manager import DeequInspectorManager
from tadv.loader import FileLoader


def test_spark_df_to_column_desc(resources_path):
    file_path = (
            resources_path
            / "example_dataset_1"
            / "files"
            / "example_table.csv"
    )
    dq_manager = DeequDataQualityManager()
    df = FileLoader.load_csv(file_path)
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)
    yaml_string = DeequInspectorManager().spark_df_to_column_desc(spark_df, spark)
    assert yaml_string.startswith("UserName:\n  completeness: 1.0\n")
