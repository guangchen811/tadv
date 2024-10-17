import os

os.environ["SPARK_VERSION"] = "3.5"
import argparse
import logging

from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.llm.langchain._model import LangChainCADV
from cadv_exploration.loader import load_csv, load_py_files
from cadv_exploration.utils import get_project_root

logging.basicConfig(
    filename="log/langchain_cadv.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    argparse.ArgumentParser(description="Run LangChainCADV")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("--script-id", type=int, help="Script ID", default=0)
    args = parser.parse_args()
    logging.info(f"Model: {args.model}")
    logging.info(f"Script ID: {args.script_id}")

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
    column_desc = spark_df_to_column_desc(spark, spark_df)

    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    scripts = load_py_files(dir_path)

    input_variables = {
        "column_desc": column_desc,
        "script": scripts[args.script_id],
    }

    lc = LangChainCADV(model=args.model)

    relevant_columns_list, expectations, rules = lc.invoke(
        input_variables=input_variables
    )

    logging.info(f"Relevant columns: {relevant_columns_list}")
    logging.info(f"Expectations: {expectations}")
    logging.info(f"Rules: {rules}")


if __name__ == "__main__":
    main()
