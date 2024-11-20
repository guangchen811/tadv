from cadv_exploration.utils import get_current_folder
from cadv_exploration.utils import load_dotenv

load_dotenv()

import pytest
from cadv_exploration.deequ_wrapper import DeequWrapper


@pytest.fixture
def deequ_wrapper():
    return DeequWrapper()


@pytest.fixture
def resources_path():
    return get_current_folder() / "resources"
