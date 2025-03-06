from tadv.data_models import CodeEntry, ColumnConstraints, \
    Constraints
from tadv.utils import get_current_folder
from tadv.utils import load_dotenv

load_dotenv()

import pytest
from tadv.dq_manager import DeequDataQualityManager


@pytest.fixture
def dq_manager():
    return DeequDataQualityManager()


@pytest.fixture
def resources_path():
    return get_current_folder() / "resources"


@pytest.fixture
def constraints_instance():
    code_entries = [
        CodeEntry(suggestion="Use a non-null constraint", validity="Valid"),
        CodeEntry(suggestion="Ensure unique values", validity="Invalid"),
    ]
    column_constraints = ColumnConstraints(code=code_entries, assumptions=["Assumption 1", "Assumption 2"])
    constraints = Constraints(constraints={"column1": column_constraints})
    return constraints
