from cadv_exploration.runtime_environments import PythonExecutor
from cadv_exploration.utils import get_project_root


def run_ml_inference(processed_idx, dataset_name, downstream_task_type, single_script=""):
    timeout = 6000
    executor = PythonExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / f"{dataset_name}"
    processed_data_path = project_root / "data_processed" / f"{dataset_name}_ml_inference_{downstream_task_type}" / processed_idx
    script_dir = original_data_path / "scripts" / "ml"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        if not script_path.stem.startswith(f"{downstream_task_type}"):
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
    dataset_name = "playground-series-s4e10"
    downstream_task_type = "regression"
    run_ml_inference(processed_idx="base_version", dataset_name=dataset_name, downstream_task_type=downstream_task_type,
                     single_script="regression_4.py")
