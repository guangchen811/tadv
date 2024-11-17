from cadv_exploration.utils import load_dotenv

load_dotenv()
import argparse
import logging
import oyaml as yaml

from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.deequ import get_suggestion_for_spark_df
from cadv_exploration.loader import load_csv
from cadv_exploration.utils import get_project_root

from scripts.python.constraint_inference.utils import filter_constraints

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler with write-plus mode
file_handler = logging.FileHandler("deequ.log", mode="a")
file_handler.setLevel(logging.INFO)

# Define the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


def main():
    argparse.ArgumentParser(description="Run Deequ")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    local_project_path = get_project_root() / "data" / "playground-series-s4e10"
    train_file_path = local_project_path / "files_with_clean_test_data" / "train.csv"
    validation_file_path = train_file_path.parent.parent / "files_with_clean_test_data" / "validation.csv"

    output_path = local_project_path / "output" / "deequ_constraints.yaml"

    train_data = load_csv(train_file_path)
    validation_data = load_csv(validation_file_path)

    spark_train_data, spark_train = spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = spark_df_from_pandas_df(validation_data)

    suggestion = get_suggestion_for_spark_df(spark_train_data, spark_train)
    code_list_for_constraints = [item["code_for_constraint"] for item in suggestion]
    columns_set = set([item["column_name"] for item in suggestion])
    code_list_for_constraints_valid = filter_constraints(code_list_for_constraints, spark_validation,
                                                         spark_validation_data, logger)

    yaml_dict = {"constraints": {f"{column}": {"code": [], "assumptions": []} for column in
                                 columns_set}}
    for item in suggestion:
        code = item["code_for_constraint"]
        column_name = item["column_name"]
        if code in code_list_for_constraints_valid:
            yaml_dict["constraints"][column_name]["code"].append([code, "Valid"])
        else:
            yaml_dict["constraints"][column_name]["code"].append([code, "Invalid"])

    with open(output_path, "w") as file:
        yaml.dump(yaml_dict, file)

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    main()
