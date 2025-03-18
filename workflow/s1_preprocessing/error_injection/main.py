from tadv.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager
from tadv.error_injection.managers.sql_query import GeneralErrorInjectionManager
from tadv.utils import get_project_root


def error_injection(dataset_name, downstream_task, error_config_file_name):
    project_root = get_project_root()
    raw_file_path = project_root / "data" / dataset_name / "files"
    target_table_name = "healthcare_dataset" if dataset_name == "healthcare_dataset" else "train"
    processed_data_dir = project_root / "data_processed" / dataset_name / downstream_task

    if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
        if downstream_task == "ml_inference_classification":
            target_table_name = target_table_name
            target_column_name = "Test Results" if dataset_name == "healthcare_dataset" else "loan_status"
            submission_default_value = "Normal"
        elif downstream_task == "ml_inference_regression":
            target_table_name = target_table_name
            target_column_name = "Billing Amount" if dataset_name == "healthcare_dataset" else "person_income"
            submission_default_value = 0
        else:
            raise ValueError("Unknown downstream task")
        error_injection_manager = MLInferenceErrorInjectionManager(
            raw_file_path=raw_file_path,
            target_table_name=target_table_name,
            target_column_name=target_column_name,
            processed_data_dir=processed_data_dir,
            submission_default_value=submission_default_value
        )
    elif downstream_task in ["sql_query", "webpage_generation"]:
        error_injection_manager = GeneralErrorInjectionManager(
            raw_file_path=raw_file_path,
            target_table_name=target_table_name,
            processed_data_dir=processed_data_dir,
            sample_size=1.0
        )
    else:
        raise ValueError(f"Downstream task {downstream_task} is not supported.")

    corrupts = error_injection_manager.load_error_injection_config(
        error_injection_config_path=project_root / "data" / dataset_name / "errors" / error_config_file_name)

    error_injection_manager.error_injection(corrupts)

    # Save the corrupted test data
    error_injection_manager.save_data()

    print(f"Error injection for {dataset_name} is done.\nThe corrupted data is saved in {processed_data_dir}")


if __name__ == "__main__":
    import argparse

    project_root = get_project_root()

    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                               "webpage_generation"]


    def parse_multiple_indices(input_str, options_list):
        """Parses comma-separated indices or 'all'."""
        if input_str.lower() == "all":
            return options_list
        indices = list(map(int, input_str.split(",")))
        return [options_list[i] for i in indices]


    parser = argparse.ArgumentParser(description='Error Injection')
    parser.add_argument('--dataset-option', type=str, default="all",
                        help='Comma-separated dataset options or "all". Options: 0: playground-series-s4e10, 1: healthcare_dataset')
    parser.add_argument('--downstream-task-option', type=str, default="all",
                        help='Comma-separated downstream task options or "all". Options: 0: ml_inference_classification, 1: ml_inference_regression, 2: sql_query, 3: webpage_generation')
    args = parser.parse_args()

    # Parse inputs
    dataset_selections = parse_multiple_indices(args.dataset_option, dataset_name_options)
    downstream_task_selections = parse_multiple_indices(args.downstream_task_option, downstream_task_options)

    # Execute error injection for each combination
    for dataset_name in dataset_selections:
        for downstream_task in downstream_task_selections:
            error_config_selections = [error_config_file.name for error_config_file in
                                       (project_root / "data" / dataset_name / "errors").iterdir()]
            for error_config_file_name in error_config_selections:
                if not error_config_file_name.startswith(downstream_task):
                    continue
                error_injection(
                    dataset_name=dataset_name,
                    downstream_task=downstream_task,
                    error_config_file_name=error_config_file_name
                )
