from cadv_exploration.utils import load_dotenv

load_dotenv()
import argparse
import logging

from cadv_exploration.deequ_wrapper import DeequWrapper
from cadv_exploration.loader import FileLoader
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

    deequ_wrapper = DeequWrapper()

    project_root = get_project_root()
    original_train_file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "broken_files"
            / "original_train"
            / "healthcare_dataset.csv"
    )
    original_validation_file_path = original_train_file_path.parent.parent / "original_validation" / "healthcare_dataset.csv"
    pre_corruption_file_path = original_train_file_path.parent.parent / "pre_corruption" / "healthcare_dataset.csv"
    post_corruption_file_path = original_train_file_path.parent.parent / "post_corruption" / "healthcare_dataset.csv"

    original_train_df = FileLoader.load_csv(original_train_file_path)
    original_validation_df = FileLoader.load_csv(original_validation_file_path)
    pre_corruption_df = FileLoader.load_csv(pre_corruption_file_path)
    post_corruption_df = FileLoader.load_csv(post_corruption_file_path)

    spark_original_train_df, spark_original_train = deequ_wrapper.spark_df_from_pandas_df(original_train_df)
    spark_original_validation_df, spark_original_validation = deequ_wrapper.spark_df_from_pandas_df(
        original_validation_df)
    spark_pre_corruption_df, spark_pre_corruption = deequ_wrapper.spark_df_from_pandas_df(pre_corruption_df)
    spark_post_corruption_df, spark_post_corruption = deequ_wrapper.spark_df_from_pandas_df(post_corruption_df)

    suggestion = deequ_wrapper.get_suggestion_for_spark_df(spark_original_train, spark_original_train_df)
    code_list_for_constraints = [item["code_for_constraint"] for item in suggestion]

    filter_constraints(code_list_for_constraints, spark_original_validation, spark_original_validation_df, logger)
    # Validate the constraints on the before broken data
    deequ_wrapper.validate_on_df(spark_pre_corruption, spark_pre_corruption_df, code_list_for_constraints)

    # Validate the constraints on the after broken data
    deequ_wrapper.validate_on_df(spark_post_corruption, spark_post_corruption_df, code_list_for_constraints)

    spark_pre_corruption.sparkContext._gateway.shutdown_callback_server()
    spark_post_corruption.sparkContext._gateway.shutdown_callback_server()
    spark_pre_corruption.stop()
    spark_post_corruption.stop()


if __name__ == "__main__":
    main()
