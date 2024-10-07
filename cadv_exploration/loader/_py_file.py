import os
from typing import List


def load_py_file(file_path: str) -> str:
    """Load a Python file into a string."""
    with open(file_path, "r") as file:
        return file.read()


def load_py_files(dir_path: str) -> List[str]:
    """Load a list of Python files into a list of strings."""
    file_path = [f"{dir_path}/{file}" for file in os.listdir(dir_path)]
    return [load_py_file(file) for file in file_path]
