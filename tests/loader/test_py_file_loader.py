import os
from pathlib import Path

from cadv_exploration.loader import load_py_file, load_py_files
from cadv_exploration.utils import get_project_root


def test_load_py_file():
    """Test the load_py_file function."""
    project_root = get_project_root()
    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    file_path = os.listdir(dir_path)[0]
    abs_file_path = dir_path / file_path
    script = load_py_file(abs_file_path)
    assert isinstance(script, str)
    assert len(script) > 0
    assert script.startswith("#!/usr/bin/env python")

    file_path_return = load_py_file(abs_file_path, return_file=False)
    assert isinstance(file_path_return, Path)
    assert file_path_return == abs_file_path


def test_load_py_files():
    """Test the load_py_files function."""
    project_root = get_project_root()
    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "kernels_py"
    scripts = load_py_files(dir_path)
    assert len(scripts) == 3
    assert all([type(script) == str for script in scripts])
    assert all([len(script) > 0 for script in scripts])
    assert all([script.startswith("#!/usr/bin/env python") for script in scripts])

    file_path_return = load_py_files(dir_path, return_files=True)
    assert isinstance(file_path_return, list)
    assert isinstance(file_path_return[0], str)
