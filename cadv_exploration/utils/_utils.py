"""
Some useful utils for the project
"""

from pathlib import Path

import dotenv


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent


def load_dotenv():
    """Load the .env file."""
    dotenv.load_dotenv(get_project_root() / ".env")
