from tadv.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager
from tadv.error_injection.managers.sql_query import GeneralErrorInjectionManager
from tadv.utils import get_project_root


def error_injection(dataset_name, downstream_task, error_config_file_name):
    assert dataset_name == "healthcare_dataset"
    project_root = get_project_root()
    raw_file_path = project_root / "data" / dataset_name / "files"
    target_table_name = "healthcare_dataset"
    processed_data_dir = project_root / "data_processed" / dataset_name / downstream_task

    if downstream_task in ["ml_inference_classification", "ml_inference_regression"]:
        if downstream_task == "ml_inference_classification":
            target_table_name = target_table_name
            target_column_name = "Test Results"
            submission_default_value = "Normal"
        elif downstream_task == "ml_inference_regression":
            target_table_name = target_table_name
            target_column_name = "Billing Amount"
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
    dataset_name = "healthcare_dataset"
    downstream_task_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                               "webpage_generation"]
    error_config_file_name_options = ["config_1.yaml"]

    error_injection(
        dataset_name=dataset_name,
        downstream_task=downstream_task_options[2],
        error_config_file_name=error_config_file_name_options[0]
    )
