from cadv_exploration.runtime_environments import PythonExecutor
from cadv_exploration.utils import get_project_root
from utils._utils import get_task_instance


def run_web_code(processed_idx, single_script=""):
    executor = PythonExecutor()
    project_root = get_project_root()
    dataset_name = "healthcare_dataset"
    original_data_path = project_root / "data" / dataset_name
    processed_data_path = project_root / "data_processed" / dataset_name / f"{processed_idx}"
    script_dir = original_data_path / "scripts" / "web"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        # task_instance = get_task_instance(script_path)
        # print(f"Running script: {task_instance.original_code}")
        output_path = processed_data_path / "output" / script_path.stem / "results_on_clean_test_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=original_data_path.name, script_path=script_path,
                            input_path=processed_data_path / "files_with_clean_test_data",
                            output_path=processed_data_path / "output" / script_path.stem / "results_on_clean_test_data")
        output_path = processed_data_path / "output" / script_path.stem / "results_on_corrupted_test_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=original_data_path.name, script_path=script_path,
                            input_path=processed_data_path / "files_with_corrupted_test_data",
                            output_path=processed_data_path / "output" / script_path.stem / "results_on_corrupted_test_data")


if __name__ == "__main__":
    run_web_code(processed_idx="base_version")