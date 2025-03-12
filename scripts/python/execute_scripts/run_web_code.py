from tadv.runtime_environments import PythonExecutor
from tadv.utils import get_project_root


def run_web_code(processed_data_label, dataset_name, single_script=""):
    executor = PythonExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / dataset_name
    processed_data_path = project_root / "data_processed" / f"{dataset_name}" / "webpage_generation" / f"{processed_data_label}"
    script_dir = original_data_path / "scripts" / "webpage_generation"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        output_path = processed_data_path / "output" / script_path.stem / "results_on_clean_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=original_data_path.name, script_path=script_path,
                     input_path=processed_data_path / "files_with_clean_new_data",
                     output_path=processed_data_path / "output" / script_path.stem / "results_on_clean_new_data")
        output_path = processed_data_path / "output" / script_path.stem / "results_on_corrupted_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=original_data_path.name, script_path=script_path,
                     input_path=processed_data_path / "files_with_corrupted_new_data",
                     output_path=processed_data_path / "output" / script_path.stem / "results_on_corrupted_new_data")


if __name__ == "__main__":
    # run_web_code(processed_data_label="base_version", dataset_name="healthcare_dataset", single_script="info_9")
    run_web_code(processed_data_label="0", dataset_name="playground-series-s4e10")
