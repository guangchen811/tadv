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

    original_train_df = load_csv(original_train_file_path)
    original_validation_df = load_csv(original_validation_file_path)
    pre_corruption_df = load_csv(pre_corruption_file_path)
    post_corruption_df = load_csv(post_corruption_file_path)

    spark_original_train_df, spark_original_train = spark_df_from_pandas_df(original_train_df)
    spark_original_validation_df, spark_original_validation = spark_df_from_pandas_df(original_validation_df)
    spark_pre_corruption_df, spark_pre_corruption = spark_df_from_pandas_df(pre_corruption_df)
    spark_post_corruption_df, spark_post_corruption = spark_df_from_pandas_df(post_corruption_df)

    suggestion = get_suggestion_for_spark_df(spark_original_train, spark_original_train_df)
    code_list_for_constraints = [item["code_for_constraint"] for item in suggestion]

    check_original = Check(spark_original_validation, CheckLevel.Warning, "Check for original data")
    check_result_on_original_validation_df = apply_checks_from_strings(check_original, code_list_for_constraints,
                                                                       spark_original_validation,
                                                                       spark_original_validation_df)
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
                                               check_result_on_original_validation_df[i] == None]
    grammarly_failure_check_output_on_original_validation_df = "\n".join(failure_check_on_original_validation_df)
    logger.info(f"Check result on original data: {check_result_on_original_validation_df}")
    logger.info(
        f"Success on original data: {success_on_original_validation_df} / {len(status_on_original_validation_df)}")
    logger.info(f"Failure check on original data: {failure_check_output_on_original_validation_df}")
    logger.info(f"Grammarly failure check on original data: {grammarly_failure_check_output_on_original_validation_df}")

    code_list_for_constraints = [code_list_for_constraints[i] for i in range(len(code_list_for_constraints)) if
                                 status_on_original_validation_df[i] == "Success"]
    logger.info(f"Code list for constraints: {code_list_for_constraints}")


    # Validate the constraints on the before broken data
    check_pre_corruption = Check(spark_pre_corruption, CheckLevel.Warning, "Check for data before corruption")
    validate_on_df(check_pre_corruption, code_list_for_constraints, spark_pre_corruption, spark_pre_corruption_df)

    # Validate the constraints on the after broken data
    check_post_corruption = Check(spark_post_corruption, CheckLevel.Warning, "Check for data after corruption")
    validate_on_df(check_post_corruption, code_list_for_constraints, spark_post_corruption, spark_post_corruption_df)

    spark_pre_corruption.sparkContext._gateway.shutdown_callback_server()
    spark_post_corruption.sparkContext._gateway.shutdown_callback_server()
    spark_pre_corruption.stop()
    spark_post_corruption.stop()

def validate_on_df(check, code_list_for_constraints, spark, spark_df):
    check_result_on_post_corruption_df = apply_checks_from_strings(check, code_list_for_constraints,
                                                                   spark, spark_df)
    status_on_post_corruption_df = [item['constraint_status'] if
                                    item is not None else None for item in check_result_on_post_corruption_df]
    success_on_post_corruption_df = status_on_post_corruption_df.count("Success")
    failure_check_on_post_corruption_df = [item.constraint for item in check_result_on_post_corruption_df if
                                           item['constraint_status'] == 'Failure']
    failure_check_output_on_post_corruption_df = "\n".join(failure_check_on_post_corruption_df)
    logger.info(f"Check result on after broken data: {check_result_on_post_corruption_df}")
    logger.info(f"Success on after broken data: {success_on_post_corruption_df} / {len(status_on_post_corruption_df)}")
    logger.info(f"Failure check on after broken data: {failure_check_output_on_post_corruption_df}")

if __name__ == "__main__":
    main()
