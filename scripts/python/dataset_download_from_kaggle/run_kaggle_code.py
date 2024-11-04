import os
import nbformat

from cadv_exploration.runtime_environments.kaggle import KaggleExecutor
from cadv_exploration.utils import get_project_root


def run_kaggle_code():
    executor = KaggleExecutor()
    project_root = get_project_root()
    local_project_path = (
            project_root
            / "data"
            / "santander-value-prediction-challenge"
    )
    notebooks = find_notebooks_with_to_csv()
    idx = 11
    script_name = notebooks[idx].split(".")[0]
    print(f"Running script: {script_name}")
    script_path = local_project_path / "kernels_ipynb" / f"{script_name}.ipynb"
    output_path = local_project_path / "output" / script_name
    result = executor.run(local_project_path=local_project_path, script_path=script_path, output_path=output_path)
    print(result)


def find_notebooks_with_to_csv():
    # List to store notebooks with .to_csv( calls
    notebooks_with_tocsv = []
    project_root = get_project_root()
    local_script_dir = (
            project_root
            / "data"
            / "santander-value-prediction-challenge"
            / "kernels_ipynb"
    )

    # Iterate through files in the directory
    for filename in os.listdir(local_script_dir):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(local_script_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
                # Check each cell in the notebook
                for cell in notebook.cells:
                    if cell.cell_type == 'code' and '.to_csv(' in cell.source:
                        notebooks_with_tocsv.append(filename)
                        break  # Found .to_csv, no need to check further cells in this notebook
    return notebooks_with_tocsv


if __name__ == "__main__":
    run_kaggle_code()
