from pathlib import Path

from cadv_exploration.loader import FileLoader


def test_load_py_file(resources_path):
    """Test the load_py_file function."""
    file_path = (resources_path
                 / "example_dataset_1"
                 / "kernel_py"
                 / "example_notebook.py"
                 )
    script = FileLoader.load_py_file(file_path)
    assert isinstance(script, str)
    assert len(script) > 0
    assert script.startswith("#%% md")

    file_path_return = FileLoader.load_py_file(file_path, return_file=False)
    assert isinstance(file_path_return, Path)
    assert file_path_return == file_path


def test_load_py_files(resources_path):
    """Test the load_py_files function."""
    dir_path = (resources_path
                / "example_dataset_1"
                / "kernel_py"
                )
    scripts = FileLoader.load_py_files(dir_path)
    assert len(scripts) == 1
    assert all([type(script) == str for script in scripts])
    assert all([len(script) > 0 for script in scripts])
    assert all([script.startswith("#%% md") for script in scripts])

    file_path_return = FileLoader.load_py_files(dir_path, return_files=True)
    assert isinstance(file_path_return, list)
    assert isinstance(file_path_return[0], str)
