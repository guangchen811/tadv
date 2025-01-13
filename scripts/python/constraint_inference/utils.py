import argparse
import logging

from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


def filter_constraints(code_list_for_constraints, spark_original_validation, spark_original_validation_df, logger):
    dq_manager = DeequDataQualityManager()

    logger.info(f"Suggested Code list for constraints: {code_list_for_constraints}")
    check_result_on_original_validation_df = dq_manager.apply_checks_from_strings(spark_original_validation,
                                                                                  spark_original_validation_df,
                                                                                  code_list_for_constraints)
    status_on_original_validation_df = [item['constraint_status'] if
                                        item is not None else None for item in check_result_on_original_validation_df]
    success_on_original_validation_df = status_on_original_validation_df.count("Success")
    failure_check_on_original_validation_df = [code_list_for_constraints[i] for i in
                                               range(len(check_result_on_original_validation_df)) if
                                               check_result_on_original_validation_df[i] is not None and
                                               check_result_on_original_validation_df[i][
                                                   'constraint_status'] == 'Failure']
    failure_check_output_on_original_validation_df = "\n".join(failure_check_on_original_validation_df)
    failure_check_on_original_validation_df = [code_list_for_constraints[i] for i in
                                               range(len(code_list_for_constraints)) if
                                               check_result_on_original_validation_df[i] is None]
    grammarly_failure_check_output_on_original_validation_df = "\n".join(failure_check_on_original_validation_df)
    logger.info(f"Check result on original data: {check_result_on_original_validation_df}")
    logger.info(
        f"Success on original data: {success_on_original_validation_df} / {len(status_on_original_validation_df)}")
    logger.info(f"Failure check on original data: {failure_check_output_on_original_validation_df}")
    logger.info(f"Grammarly failure check on original data: {grammarly_failure_check_output_on_original_validation_df}")
    # remove the constraints that are not grammarly correct
    code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                 status_on_original_validation_df[i] == "Success"]
    logger.info(f"Filtered Code list for constraints: {code_list_for_constraints}")
    return code_list_for_constraints


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    """Set up the logger with file handler and formatter."""
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_arguments(description: str) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, help="Maximum number of retries", default=3)
    return parser.parse_args()


def load_train_and_test_spark_data(data_name: str, processed_data_idx: int, dq_manager) -> tuple:
    """
    Load train and validation CSV files into Spark dataframes.

    Returns:
        spark_train_data, spark_train, spark_validation_data, spark_validation
    """
    processed_data_path = get_project_root() / "data_processed" / f"{data_name}" / f"{processed_data_idx}"
    train_file_path = processed_data_path / "files_with_clean_test_data" / "train.csv"
    validation_file_path = processed_data_path / "files_with_clean_test_data" / "validation.csv"

    train_data = FileLoader.load_csv(train_file_path)
    validation_data = FileLoader.load_csv(validation_file_path)

    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    spark_validation_data, spark_validation = dq_manager.spark_df_from_pandas_df(validation_data)
    return spark_train_data, spark_train, spark_validation_data, spark_validation
