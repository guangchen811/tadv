from cadv_exploration.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager

from cadv_exploration.utils import get_project_root


def error_injection():
    project_root = get_project_root()
    error_injection_manager = MLInferenceErrorInjectionManager(
        raw_file_path=project_root / "data" / "healthcare_dataset" / "files",
        target_table_name="healthcare_dataset",
        target_column_name="Test Results",
        processed_data_dir=project_root / "data_processed" / "healthcare_dataset_ml_inference",
        submission_default_value="Normal",
    )

    # Inject errors on the test data
    corrupts = build_corrupts()

    error_injection_manager.error_injection(corrupts)

    # Save the corrupted test data
    error_injection_manager.save_data()


def build_corrupts():
    corrupts = []
    return corrupts


if __name__ == "__main__":
    error_injection()
