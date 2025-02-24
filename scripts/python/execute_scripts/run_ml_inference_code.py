from cadv_exploration.runtime_environments import PythonExecutor
from cadv_exploration.utils import get_project_root


def run_ml_inference(processed_idx, single_script=""):
    timeout = 6000
    executor = PythonExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / "healthcare_dataset"
    processed_data_path = project_root / "data_processed" / "healthcare_dataset_ml_inference" / f"{processed_idx}"
    script_dir = original_data_path / "scripts" / "ml"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        output_path = processed_data_path / "output" / script_path.stem / "results_on_clean_test_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=processed_data_path.parent.name,
                     script_path=script_path,
                     input_path=processed_data_path / "files_with_clean_test_data",
                     output_path=output_path,
                     timeout=timeout)
        output_path = processed_data_path / "output" / script_path.stem / "results_on_corrupted_test_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=processed_data_path.parent.name,
                     script_path=script_path,
                     input_path=processed_data_path / "files_with_corrupted_test_data",
                     output_path=output_path,
                     timeout=timeout)


if __name__ == "__main__":
    # run_ml_inference(processed_idx="base_version")
    run_ml_inference(processed_idx="base_version", single_script="classification_5.py")
