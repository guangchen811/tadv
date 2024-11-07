from cadv_exploration.utils import load_dotenv

load_dotenv()
import argparse
import logging

from pydeequ import Check, CheckLevel
from cadv_exploration.deequ import spark_df_from_pandas_df
from cadv_exploration.deequ import get_suggestion_for_spark_df, apply_checks_from_strings
from cadv_exploration.loader import load_csv
from cadv_exploration.utils import get_project_root

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

    project_root = get_project_root()
    original_file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "broken_files"
            / "original"
            / "healthcare_dataset.csv"
    )
    before_broken_file_path = original_file_path.parent.parent / "before_broken" / "healthcare_dataset.csv"
    after_broken_file_path = original_file_path.parent.parent / "after_broken" / "healthcare_dataset.csv"

    original_df = load_csv(original_file_path)
    before_broken_df = load_csv(before_broken_file_path)
    after_broken_df = load_csv(after_broken_file_path)

    spark_original_df, spark_original = spark_df_from_pandas_df(original_df)
    spark_before_broken_df, spark_before_broken = spark_df_from_pandas_df(before_broken_df)
    spark_after_broken_df, spark_after_broken = spark_df_from_pandas_df(after_broken_df)

    suggestion = get_suggestion_for_spark_df(spark_original, spark_original_df)
    code_list_for_constraints = [item["code_for_constraint"] for item in suggestion if
                                 "Room Number" not in item["code_for_constraint"]]

    logger.info(f"Code list for constraints: {code_list_for_constraints}")

    check_before_broken = Check(spark_before_broken, CheckLevel.Warning, "Check for data before broken")
    check_after_broken = Check(spark_after_broken, CheckLevel.Warning, "Check for data after broken")

    check_result_on_before_broken_df = apply_checks_from_strings(check_before_broken, code_list_for_constraints,
                                                                 spark_before_broken, spark_before_broken_df)
    status_on_before_broken_df = [item['constraint_status'] for item in check_result_on_before_broken_df if item is not None]
    success_on_before_broken_df = status_on_before_broken_df.count("Success")
    failure_check_on_before_broken_df = [item.constraint for item in check_result_on_before_broken_df if
                                         item['constraint_status'] == 'Failure']
    failure_check_output_on_before_broken_df = "\n".join(failure_check_on_before_broken_df)
    logger.info(f"Check result on before broken data: {check_result_on_before_broken_df}")
    logger.info(f"Success on before broken data: {success_on_before_broken_df} / {len(status_on_before_broken_df)}")
    logger.info(f"Failure check on before broken data: {failure_check_output_on_before_broken_df}")

    check_result_on_after_broken_df = apply_checks_from_strings(check_after_broken, code_list_for_constraints,
                                                                spark_after_broken, spark_after_broken_df)
    status_on_after_broken_df = [item['constraint_status'] for item in check_result_on_after_broken_df if item is not None]
    success_on_after_broken_df = status_on_after_broken_df.count("Success")
    failure_check_on_after_broken_df = [item.constraint for item in check_result_on_after_broken_df if
                                        item['constraint_status'] == 'Failure']
    failure_check_output_on_after_broken_df = "\n".join(failure_check_on_after_broken_df)
    logger.info(f"Check result on after broken data: {check_result_on_after_broken_df}")
    logger.info(f"Success on after broken data: {success_on_after_broken_df} / {len(status_on_after_broken_df)}")
    logger.info(f"Failure check on after broken data: {failure_check_output_on_after_broken_df}")

    spark_before_broken.sparkContext._gateway.shutdown_callback_server()
    spark_after_broken.sparkContext._gateway.shutdown_callback_server()
    spark_before_broken.stop()
    spark_after_broken.stop()


if __name__ == "__main__":
    main()
