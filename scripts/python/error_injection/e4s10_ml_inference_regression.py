from cadv_exploration.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager

from cadv_exploration.utils import get_project_root


# This case can be treated as context information for ml inference. (the test data needs to be validated)
def error_injection():
    project_root = get_project_root()
    error_injection_manager = MLInferenceErrorInjectionManager(
        raw_file_path=project_root / "data" / "playground-series-s4e10" / "files",
        target_table_name="train",
        target_column_name="person_income",
        processed_data_dir=project_root / "data_processed" / "playground-series-s4e10_ml_inference_regression",
        submission_default_value=0,
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
