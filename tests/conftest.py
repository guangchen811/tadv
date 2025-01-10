from cadv_exploration.utils import get_current_folder
from cadv_exploration.utils import load_dotenv

load_dotenv()

import pytest
from cadv_exploration.dq_manager import DeequDataQualityManager


@pytest.fixture
def dq_manager():
    return DeequDataQualityManager()


@pytest.fixture
def resources_path():
    return get_current_folder() / "resources"
