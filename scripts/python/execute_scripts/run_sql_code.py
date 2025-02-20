from cadv_exploration.runtime_environments import DuckDBExecutor
from cadv_exploration.utils import get_project_root
from utils._utils import get_task_instance


def run_sql_code(processed_idx, single_script=""):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / "healthcare_dataset"
    processed_data_path = project_root / "data_processed" / "healthcare_dataset_sql_query" / f"{processed_idx}"
    script_dir = original_data_path / "scripts" / "sql"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        task_instance = get_task_instance(script_path)
        print(f"Running script: {task_instance.original_script}")
        output_path = processed_data_path / "output" / script_path.stem / "results_on_clean_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run_script(project_name=original_data_path.name, script_context=task_instance.original_script,
                            input_path=processed_data_path / "files_with_clean_new_data",
                            output_path=processed_data_path / "output" / script_path.stem / "results_on_clean_new_data")
        output_path = processed_data_path / "output" / script_path.stem / "results_on_corrupted_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run_script(project_name=original_data_path.name, script_context=task_instance.original_script,
                            input_path=processed_data_path / "files_with_corrupted_new_data",
                            output_path=processed_data_path / "output" / script_path.stem / "results_on_corrupted_new_data")


if __name__ == "__main__":
    run_sql_code(processed_idx="base_version")
    # run_sql_code(processed_idx="0", single_script="dev_5")
