import argparse
import logging

from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


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
