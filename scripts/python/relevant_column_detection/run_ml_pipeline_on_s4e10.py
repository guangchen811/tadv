from cadv_exploration.utils import load_dotenv

load_dotenv()

import argparse
import importlib.util
import logging

from deequ_wrapper import DeequWrapper
from inspector.deequ._to_string import spark_df_to_column_desc
from llm.langchain import LangChainCADV
from llm.langchain._downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from loader import FileLoader
from utils import get_project_root

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler with write-plus mode
file_handler = logging.FileHandler("langchain_cadv_relevant_column_inference.log", mode="a")
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
    logger.info(f"Model: {args.model}")

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

    scripts_path_dir = original_data_path / "scripts_ml"

    for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
        if not script_path.name.endswith(".py"):
            continue
        result_path = processed_data_path / "constraints" / f"{script_path.name.split('.')[0]}" / "cadv_constraints.yaml"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        module_name = script_path.stem
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        task_class = getattr(module, "KaggleLoanColumnDetectionTask")
        task_instance = task_class()
        script_context = task_instance.original_code
        input_variables = {
            "column_desc": column_desc,
            "script": script_context,
        }

        lc = LangChainCADV(model=args.model, downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

        max_retries = args.max_retries
        relevant_columns_list, expectations, suggestions = lc.invoke(
            input_variables=input_variables, num_stages=1, max_retries=max_retries
        )
        print(module_name)
        ground_truth = sorted(task_instance.target_columns(), key=lambda x: x.lower())
        relevant_columns_list = sorted(relevant_columns_list, key=lambda x: x.lower())
        print(f"Ground Truth:\n{ground_truth}")
        print(f"Relevant Columns:\n{relevant_columns_list}")
        print('')


if __name__ == "__main__":
    run_langchain_cadv(7)
