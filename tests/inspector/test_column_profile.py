from cadv_exploration.utils import load_dotenv

load_dotenv()
from cadv_exploration.deequ_wrapper import DeequWrapper
from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


def test_spark_df_to_column_desc():
    project_root = get_project_root()
    file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "files"
            / "healthcare_dataset.csv"
    )
    deequ_wrapper = DeequWrapper()
    df = FileLoader.load_csv(file_path)
    spark_df, spark = deequ_wrapper.spark_df_from_pandas_df(df)
    yaml_string = spark_df_to_column_desc(spark_df, spark)
    print(yaml_string)
