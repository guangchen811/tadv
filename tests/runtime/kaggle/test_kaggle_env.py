import os
import subprocess
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from tadv.runtime_environments import KaggleExecutor
from tadv.utils import get_project_root


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
def test_runnable_on_small_test_dataset_with_ipynb(tmp_path):
    executor = KaggleExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    input_path = Path(
        local_project_path / "files"
    )
    script_dir = Path(
        local_project_path
        / "kernel_ipynb"
        / "example_notebook.ipynb"
    )
    output_path = tmp_path / "output"
    _ = executor.run(local_project_path.name, input_path, script_dir, output_path)

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
