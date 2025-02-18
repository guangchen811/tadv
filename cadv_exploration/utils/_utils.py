"""
Some useful utils for the project
"""
import inspect
from pathlib import Path

import dotenv
import oyaml as yaml
from attr import dataclass


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


@dataclass
class TaskInstance:
    original_script: str
    script_path: Path
    annotations: dict


def get_task_instance(script_path):
    config_file_path = script_path.parent.parent.parent / "annotations" / script_path.parent.stem / f"{script_path.stem}.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    task_instance = TaskInstance(original_script=script_path.read_text(), script_path=script_path,
                                 annotations=config["annotations"])
    return task_instance
