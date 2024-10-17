import os
from typing import List


def load_py_file(file_path: str, return_file: bool = True) -> str:
    """Load a Python file into a string."""
    if return_file:
        with open(file_path, "r") as file:
            return file.read()
    else:
        return file_path


def load_py_files(dir_path: str, return_files: bool = True) -> List[str]:
    """Load a list of Python files into a list of strings."""
    file_path = [f"{dir_path}/{file}" for file in os.listdir(dir_path)]
    return [load_py_file(file, return_files) for file in file_path]
