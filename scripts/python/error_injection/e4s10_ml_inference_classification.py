from cadv_exploration.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager
from cadv_exploration.utils import get_project_root


# This case can be treated as context information for ml inference. (the test data needs to be validated)
def error_injection(error_config_file_name):
    project_root = get_project_root()
    error_injection_manager = MLInferenceErrorInjectionManager(
        raw_file_path=project_root / "data" / "playground-series-s4e10" / "files",
        target_table_name="train",
        target_column_name="loan_status",
        processed_data_dir=project_root / "data_processed" / "playground-series-s4e10_ml_inference_classification",
        submission_default_value=0.5,
    )

    corrupts = error_injection_manager.load_error_injection_config(
        error_injection_config_path=project_root / "data" / "playground-series-s4e10" / "errors" / error_config_file_name)

    error_injection_manager.error_injection(corrupts)

    # Save the corrupted test data
    error_injection_manager.save_data()


if __name__ == "__main__":
    error_config_file_name = "error_injection_config.yaml"
    error_injection(
        error_config_file_name=error_config_file_name
    )
