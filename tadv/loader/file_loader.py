from ._csv import load_csv, load_csvs
from ._py_file import load_py_file, load_py_files


class FileLoader:
    """
    A generic file loader class to handle loading different types of files.
    """

    def __init__(self):
        pass

    @classmethod
    def load_csv(cls, file_path, **kwargs):
        return load_csv(file_path, **kwargs)

    @classmethod
    def load_csvs(cls, dir_path):
        return load_csvs(dir_path)

    @classmethod
    def load_py_file(cls, file_path, return_file: bool = True):
        return load_py_file(file_path, return_file)

    @classmethod
    def load_py_files(cls, dir_path, return_files: bool = True):
        return load_py_files(dir_path, return_files)
