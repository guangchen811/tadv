from tadv.runtime_environments import DuckDBExecutor
from tadv.utils import get_project_root
from tadv.utils import get_task_instance


def run_sql_code(dataset_name, processed_data_label, single_script=""):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / f"{dataset_name}"
    processed_data_path = project_root / "data_processed" / f"{dataset_name}" / "sql_query" / f"{processed_data_label}"
    script_dir = original_data_path / "scripts" / "sql_query"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        task_instance = get_task_instance(script_path)
        output_path = processed_data_path / "output" / script_path.stem / "results_on_clean_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run_script(project_name=original_data_path.name, script_context=task_instance.original_script,
                            input_path=processed_data_path / "files_with_clean_new_data",
                            output_path=output_path)
        output_path = processed_data_path / "output" / script_path.stem / "results_on_corrupted_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run_script(project_name=original_data_path.name, script_context=task_instance.original_script,
                            input_path=processed_data_path / "files_with_corrupted_new_data",
                            output_path=output_path)


if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]

    dataset_option = 0
    processed_data_label = "0"
    single_script = ""

    run_sql_code(dataset_name=dataset_name_options[dataset_option],
                 processed_data_label=f"{processed_data_label}",
                 single_script=single_script)
