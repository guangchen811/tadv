"""
Some useful utils for the project
"""

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
