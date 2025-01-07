from cadv_exploration.runtime_environments import DuckDBExecutor
from cadv_exploration.utils import get_project_root


def run_sql_code(processed_idx, single_script=""):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / "playground-series-s4e10"
    processed_data_path = project_root / "data_processed" / "playground-series-s4e10" / f"{processed_idx}"
    script_dir = original_data_path / "scripts_sql"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=True):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        print(f"Running script: {script_path.name}")
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
    run_sql_code(processed_idx=8, single_script="feature_engineering_10")
