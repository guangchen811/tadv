from nbconvert import PythonExporter

from cadv_exploration.utils import load_dotenv

load_dotenv()
import argparse
import logging
import oyaml as yaml

from pydeequ import Check, CheckLevel
from cadv_exploration.deequ._constraint_validation import validate_on_df
from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.inspector.deequ._to_string import spark_df_to_column_desc
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root
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


def main():
    argparse.ArgumentParser(description="Run LangChainCADV")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("--script-id", type=int, help="Script ID", default=2)
    args = parser.parse_args()
    logging.info(f"Model: {args.model}")
    logging.info(f"Script ID: {args.script_id}")

    local_project_path = get_project_root() / "data" / "playground-series-s4e10"
    train_file_path = local_project_path / "files_with_clean_test_data" / "train.csv"
    validation_file_path = train_file_path.parent.parent / "files_with_clean_test_data" / "validation.csv"

    train_data = FileLoader.load_csv(train_file_path)
    validation_data = FileLoader.load_csv(validation_file_path)

    spark_train_data, spark_train = spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = spark_df_from_pandas_df(validation_data)

    column_desc = spark_df_to_column_desc(spark_train_data, spark_train)

    scripts_path_dir = local_project_path / "kernels_ipynb_selected"
    export = PythonExporter()
    for script_path in scripts_path_dir.iterdir():
        output_path = local_project_path / "output" / f"{script_path.name.split('.')[0]}" / "cadv_constraints.yaml"
        if not script_path.name.endswith(".ipynb"):
            continue
        if output_path.exists():
            continue
        script_context = export.from_filename(script_path)[0]

        input_variables = {
            "column_desc": column_desc,
            "script": script_context,
        }

        lc = LangChainCADV(model=args.model)

        relevant_columns_list, expectations, suggestions = lc.invoke(
            input_variables=input_variables
        )

        logger.info(f"Relevant columns: {relevant_columns_list}")
        logger.info(f"Expectations: {expectations}")
        logger.info(f"Suggestions: {suggestions}")

        code_list_for_constraints = [item for v in suggestions.values() for item in v]

        # Validate the constraints on the original data to see if they are grammarly correct
        code_list_for_constraints_valid = filter_constraints(code_list_for_constraints, spark_validation,
                                                             spark_validation_data, logger)
        yaml_dict = {"constraints": {f"{relevant_column}": {"code": [], "assumptions": []} for relevant_column in
                                     relevant_columns_list}}
        for suggested_column, suggestions in suggestions.items():
            if suggested_column not in relevant_columns_list:
                continue
            for suggestion in suggestions:
                if suggestion in code_list_for_constraints_valid:
                    yaml_dict["constraints"][suggested_column]["code"].append([suggestion, "Valid"])
                else:
                    yaml_dict["constraints"][suggested_column]["code"].append([suggestion, "Invalid"])
        for suggested_column, expectations in expectations.items():
            if suggested_column not in relevant_columns_list:
                continue
            for expectation in expectations:
                yaml_dict["constraints"][suggested_column]["assumptions"].append(expectation)

        with open(output_path, "w") as f:
            yaml.dump(yaml_dict, f)
    # # Validate the constraints on the before broken data
    # validate_on_df(code_list_for_constraints, spark_pre_corruption, spark_pre_corruption_df, logger)
    #
    # # Validate the constraints on the after broken data
    # validate_on_df(code_list_for_constraints, spark_post_corruption, spark_post_corruption_df, logger)
    #
    # spark_pre_corruption.sparkContext._gateway.shutdown_callback_server()
    # spark_post_corruption.sparkContext._gateway.shutdown_callback_server()
    # spark_pre_corruption.stop()
    # spark_post_corruption.stop()


if __name__ == "__main__":
    main()
