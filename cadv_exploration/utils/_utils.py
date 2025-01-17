"""
Some useful utils for the project
"""
import importlib.util
import inspect
from pathlib import Path

import dotenv


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent


def get_current_folder() -> Path:
    """
    Returns the directory where the calling script is stored.
    """
    # Get the file path of the script that called this function
    caller_file = inspect.stack()[1].filename
    # Get the directory of the caller script
    return Path(caller_file).parent


def load_dotenv():
    """Load the .env file."""
    dotenv.load_dotenv(get_project_root() / ".env")


def get_task_instance(script_path):
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_class = getattr(module, "ColumnDetectionTask")
    task_instance = task_class()
    return task_instance
