import subprocess
from pathlib import Path
import os
import pytest
import nbformat
import pandas as pd

from cadv_exploration.loader._py_file import load_py_files
from cadv_exploration.runtime_environments.kaggle import KaggleExecutor
from cadv_exploration.utils import get_project_root


def docker_image_exists(image_name="kaggle-env/python:1.0.0"):
    """
    Check if the specified Docker image exists locally.

    Args:
    - image_name (str): The name and tag of the Docker image to check.

    Returns:
    - bool: True if the image exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not docker_image_exists(), reason="Docker is not available")
def test_runnable_on_personal_dataset(tmp_path):
    executor = KaggleExecutor()
    project_root = get_project_root()
    data_path = Path(
        project_root
        / "data"
        / "prasad22"
        / "healthcare-dataset"
        / "files"
    )
    script_dir = Path(
        project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    )
    output_path = tmp_path / "output"
    script_paths = load_py_files(script_dir, return_files=False)
    result = executor.run(data_path.parent, script_paths[0], output_path)
    assert result is not None


@pytest.mark.skipif(not docker_image_exists(), reason="Docker is not available")
def test_runnable_on_competition_dataset(tmp_path):
    executor = KaggleExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root
        / "data"
        / "santander-value-prediction-challenge"
    )
    script_dir = Path(
        local_project_path / "kernels_py"
    )
    script_paths = load_py_files(script_dir, return_files=False)
    output_path = tmp_path / "output"
    result = executor.run(local_project_path, script_paths[0], output_path)
    assert result is not None


@pytest.mark.skipif(not docker_image_exists(), reason="Docker is not available")
def test_runnable_on_personal_dataset_with_ipynb(tmp_path):
    executor = KaggleExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root
        / "data"
        / "prasad22"
        / "healthcare-dataset"
    )
    script_dir = Path(
        local_project_path / "kernels_ipynb"
    )
    output_path = tmp_path / "output"
    script_paths = load_py_files(script_dir, return_files=False)
    result = executor.run(local_project_path, script_paths[0], output_path)
    assert result is not None


@pytest.mark.skipif(not docker_image_exists(), reason="Docker is not available")
def test_runnable_on_competition_dataset_with_ipynb(tmp_path):
    executor = KaggleExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root
        / "data"
        / "santander-value-prediction-challenge"
    )
    script_dir = Path(
        local_project_path / "kernels_ipynb"
    )
    script_paths = load_py_files(script_dir, return_files=False)
    output_path = tmp_path / "output"
    result = executor.run(local_project_path, script_paths[0], output_path)
    assert result is not None


@pytest.mark.skipif(not docker_image_exists(), reason="Docker is not available")
def test_runnable_on_small_test_dataset_with_ipynb(tmp_path):
    executor = KaggleExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    script_dir = Path(
        local_project_path
        / "kernel_ipynb"
        / "example_notebook.ipynb"
    )
    output_path = tmp_path / "output"
    _ = executor.run(local_project_path, script_dir, output_path)

    assert os.path.exists(output_path)
    assert len(list(output_path.iterdir())) == 1
    assert (output_path / "example_notebook.ipynb").exists()
    with open(output_path / "example_notebook.ipynb", "r") as f:
        nb = nbformat.read(f, as_version=4)
        assert len(nb.cells) == 5
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "code"
        assert nb.cells[2]['outputs'][0][
                   'text'] == "Index(['UserName', 'Identifier', 'FirstName', 'LastName'], dtype='object')\n"
    submission_file = script_dir.parent / "submission.csv"
    assert submission_file.exists()
    try:
        with open(submission_file, "r") as f:
            submission = pd.read_csv(f)
            assert 'FullName' in submission.columns
    finally:
        # Clean up: ensure that the submission.csv file is deleted
        if submission_file.exists():
            os.remove(submission_file)
