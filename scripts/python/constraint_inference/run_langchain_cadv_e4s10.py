from nbconvert import PythonExporter

from cadv_exploration.utils import load_dotenv

load_dotenv()
import argparse
import logging

from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.deequ_wrapper import DeequWrapper
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root
from cadv_exploration.data_models import Constraints
from scripts.python.constraint_inference.utils import filter_constraints

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler with write-plus mode
file_handler = logging.FileHandler("langchain_cadv.log", mode="a")
file_handler.setLevel(logging.INFO)

# Define the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


def run_langchain_cadv(processed_data_idx):
    argparse.ArgumentParser(description="Run LangChainCADV")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4o")
    parser.add_argument("--max-retries", type=int, help="Maximum number of retries", default=3)
    args = parser.parse_args()
    logging.info(f"Model: {args.model}")

    deequ_wrapper = DeequWrapper()

    original_data_path = get_project_root() / "data" / "playground-series-s4e10"
    processed_data_path = get_project_root() / "data_processed" / "playground-series-s4e10" / f"{processed_data_idx}"
    train_file_path = processed_data_path / "files_with_clean_test_data" / "train.csv"
    validation_file_path = train_file_path.parent.parent / "files_with_clean_test_data" / "validation.csv"

    train_data = FileLoader.load_csv(train_file_path)
    validation_data = FileLoader.load_csv(validation_file_path)

    spark_train_data, spark_train = deequ_wrapper.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = deequ_wrapper.spark_df_from_pandas_df(validation_data)

    column_desc = spark_df_to_column_desc(spark_train_data, spark_train)

    scripts_path_dir = original_data_path / "kernels_ipynb_selected"
    export = PythonExporter()
    for script_path in scripts_path_dir.iterdir():
        if not script_path.name.endswith(".ipynb"):
            continue
        result_path = processed_data_path / "constraints" / f"{script_path.name.split('.')[0]}" / "cadv_constraints.yaml"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        script_context = export.from_filename(script_path)[0]

        input_variables = {
            "column_desc": column_desc,
            "script": script_context,
        }

        lc = LangChainCADV(model=args.model)

        max_retries = args.max_retries
        relevant_columns_list, expectations, suggestions = lc.invoke(
            input_variables=input_variables, num_stages=3, max_retries=max_retries
        )

        code_list_for_constraints = [item for v in suggestions.values() for item in v]

        # Validate the constraints on the original data to see if they are grammarly correct
        code_list_for_constraints_valid = filter_constraints(code_list_for_constraints, spark_validation,
                                                             spark_validation_data, logger)
        constraints = Constraints.from_llm_output(relevant_columns_list, expectations, suggestions,
                                                  code_list_for_constraints_valid)

        constraints.save_to_yaml(result_path)
        print(f"Saved to {result_path}")

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    run_langchain_cadv(processed_data_idx=7)
