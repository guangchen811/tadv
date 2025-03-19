from tadv.utils import load_dotenv

load_dotenv()

from tadv.dq_manager import DeequDataQualityManager
from tadv.utils import get_project_root

from workflow.s2_experiments.utils import setup_logger, load_train_and_test_spark_data, load_previous_and_new_spark_data


def run_deequ_dv(dataset_name, downstream_task, processed_data_label):
    logger = setup_logger(get_project_root() / "logs" / "deequ_dv.log")
    dq_manager = DeequDataQualityManager()

    processed_data_path = get_project_root() / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"

    result_path = processed_data_path / "constraints" / "deequ_constraints.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
        spark_train_df, spark_train, _, _ = load_train_and_test_spark_data(
            dataset_name=dataset_name,
            downstream_task=downstream_task,
            processed_data_label=processed_data_label,
            dq_manager=dq_manager
        )
        if downstream_task == "ml_inference_classification":
            # drop the target column for classification task
            target_column_name = "Test Results" if dataset_name == "healthcare_dataset" else "loan_status"
            spark_train_df = spark_train_df.drop(target_column_name)
        elif downstream_task == "ml_inference_regression":
            # drop the target column for regression task
            target_column_name = "Billing Amount" if dataset_name == "healthcare_dataset" else "person_income"
            spark_train_df = spark_train_df.drop(target_column_name)
    elif downstream_task in ["sql_query", "webpage_generation"]:
        spark_train_df, spark_train, _, _ = load_previous_and_new_spark_data(
            dataset_name=dataset_name,
            downstream_task=downstream_task,
            processed_data_label=processed_data_label,
            dq_manager=dq_manager
        )
    else:
        raise ValueError(f"Invalid downstream task: {downstream_task}")
    spark_validation, spark_validation_df = spark_train, spark_train_df

    constraints = dq_manager.get_constraints_for_spark_df(spark_train, spark_train_df, spark_validation,
                                                          spark_validation_df)
    constraints.save_to_yaml(result_path)

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    import argparse

    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                               "webpage_generation"]


    def parse_multiple_indices(input_str, options_list):
        """Parses comma-separated indices or 'all'."""
        if input_str.lower() == "all":
            return options_list
        indices = list(map(int, input_str.split(",")))
        return [options_list[i] for i in indices]


    parser = argparse.ArgumentParser(description='Run Deequ Data Validation')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Comma-separated dataset options or "all". Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="all",
                        help='Comma-separated downstream task options or "all". Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    parser.add_argument('--processed-data-label', type=str, default="0",
                        help='Version Label of the processed data')
    args = parser.parse_args()

    # Parse inputs
    dataset_selections = parse_multiple_indices(args.dataset_option, dataset_name_options)
    downstream_task_selections = parse_multiple_indices(args.downstream_task_option, downstream_task_options)

    for dataset_name in dataset_selections:
        for downstream_task in downstream_task_selections:
            run_deequ_dv(dataset_name=dataset_name,
                         downstream_task=downstream_task,
                         processed_data_label=args.processed_data_label
                         )
