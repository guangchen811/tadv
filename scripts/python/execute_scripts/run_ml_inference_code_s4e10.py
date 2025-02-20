import os

import nbformat

from cadv_exploration.runtime_environments import KaggleExecutor
from cadv_exploration.utils import get_project_root


def run_kaggle_code(processed_idx):
    executor = KaggleExecutor()
    project_root = get_project_root()
    original_data_path = project_root / "data" / "playground-series-s4e10"
    script_dir = original_data_path / "kernels_ipynb_selected"

    processed_data_path = project_root / "data_processed" / "playground-series-s4e10_ml_inference" / f"{processed_idx}"
    input_path_with_clean_test_data = processed_data_path / "files_with_clean_test_data"
    input_path_with_corrupted_test_data = processed_data_path / "files_with_corrupted_test_data"

    notebooks = find_notebooks_with_to_csv(scirpt_dir=script_dir)

    for idx in range(len(notebooks)):
        script_name = notebooks[idx].split(".")[0]

        print(f"Running script: {script_name}")
        script_path = processed_data_path / script_dir / f"{script_name}.ipynb"
        print("on clean test data")
        output_path_on_clean_test_data = processed_data_path / "output" / script_name / "results_on_clean_test_data"
        result_on_clean_test_data = executor.run(project_name=processed_data_path.parent.name,
                                                 input_path=input_path_with_clean_test_data, script_path=script_path,
                                                 output_path=output_path_on_clean_test_data)
        print("on corrupted test data")
        output_path_on_corrupted_test_data = processed_data_path / "output" / script_name / "results_on_corrupted_test_data"
        result_on_corrupted_test_data = executor.run(project_name=processed_data_path.parent.name,
                                                     input_path=input_path_with_corrupted_test_data,
                                                     script_path=script_path,
                                                     output_path=output_path_on_corrupted_test_data)


def find_notebooks_with_to_csv(scirpt_dir):
    # List to store notebooks with .to_csv( calls
    notebooks_with_tocsv = []
    # Iterate through files in the directory
    for filename in os.listdir(scirpt_dir):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(scirpt_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
                # Check each cell in the notebook
                for cell in notebook.cells:
                    if cell.cell_type == 'code' and '.to_csv(' in cell.source:
                        notebooks_with_tocsv.append(filename)
                        break  # Found .to_csv, no need to check further cells in this notebook
    return notebooks_with_tocsv


if __name__ == "__main__":
    run_kaggle_code(processed_idx="7")
