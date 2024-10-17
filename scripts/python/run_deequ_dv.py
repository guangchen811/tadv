from cadv_exploration.utils import load_dotenv

load_dotenv()
import argparse
import logging

from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.deequ._constraint_suggestion import get_suggestion_for_spark_df
from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.llm.langchain._model import LangChainCADV
from cadv_exploration.loader import load_csv, load_py_files
from cadv_exploration.utils import get_project_root

logging.basicConfig(
    filename="log/deequ.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    argparse.ArgumentParser(description="Run LangChainCADV")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

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

    suggestion = get_suggestion_for_spark_df(spark, spark_df)

    logging.info(f"Rules: {suggestion}")


if __name__ == "__main__":
    main()
