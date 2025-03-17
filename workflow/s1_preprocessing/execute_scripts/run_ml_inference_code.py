from tadv.runtime_environments import PythonExecutor
from tadv.utils import get_project_root


def run_ml_inference(dataset_name, downstream_task_type, processed_data_label, single_script: str = ""):
    timeout = 6000
    executor = PythonExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / f"{dataset_name}"
    processed_data_path = project_root / "data_processed" / f"{dataset_name}" / f"{downstream_task_type}" / processed_data_label
    print(f"processed_data_path: {processed_data_path}")
    script_dir = original_data_path / "scripts" / "ml_inference"
    for script_path in sorted(script_dir.iterdir(), key=lambda x: x.name, reverse=False):
        if len(single_script) > 0 and single_script not in script_path.name:
            continue
        if downstream_task_type.split("_")[-1] != script_path.stem.split("_")[0]:
            continue
        output_path = processed_data_path / "output" / script_path.stem / "results_on_clean_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=processed_data_path.parent.name,
                     script_path=script_path,
                     input_path=processed_data_path / "files_with_clean_new_data",
                     output_path=output_path,
                     timeout=timeout)
        output_path = processed_data_path / "output" / script_path.stem / "results_on_corrupted_new_data"
        output_path.mkdir(parents=True, exist_ok=True)
        executor.run(project_name=processed_data_path.parent.name,
                     script_path=script_path,
                     input_path=processed_data_path / "files_with_corrupted_new_data",
                     output_path=output_path,
                     timeout=timeout)


if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_type_options = ["ml_inference_classification", "ml_inference_regression"]

    dataset_option = 0
    downstream_task_option = 0
    processed_data_label = "0"
    single_script = ""

    run_ml_inference(dataset_name=dataset_name_options[dataset_option],
                     downstream_task_type=downstream_task_type_options[downstream_task_option],
                     processed_data_label=processed_data_label,
                     single_script=single_script)
