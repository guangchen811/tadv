import subprocess
from pathlib import Path

import pytest

from cadv_exploration.loader._py_file import load_py_files
from cadv_exploration.runtime_environments.kaggle import KaggleExecutor
from cadv_exploration.utils import get_project_root


def docker_exists():
    try:
        subprocess.run(
            ["docker", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not docker_exists(), reason="Docker is not available")
def test_runnable():

    executor = KaggleExecutor()
    project_root = get_project_root()
    data_path = Path(
        project_root
        / "data"
        / "prasad22"
        / "healthcare-dataset"
        / "files"
        / "healthcare_dataset.csv"
    )
    script_dir = Path(
        project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    )

    script_paths = load_py_files(script_dir, return_files=False)
    result = executor.run(data_path.parent.parent, script_paths[0])
    assert result is not None
