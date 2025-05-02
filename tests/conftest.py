from tadv.data_models import CodeEntry, ColumnConstraints, \
    Constraints, ColumnConstraintsWithSources, ConstraintsWithSources, SourceLocation, AssumptionEntry
from tadv.utils import get_current_folder, get_project_root
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


@pytest.fixture
def constraints_with_sources_instance():
    code_entries = [
        CodeEntry(suggestion="Use a non-null constraint", validity="Valid"),
        CodeEntry(suggestion="Ensure unique values", validity="Invalid"),
    ]
    source_location = [
        SourceLocation(file="file1.py", start_line=1, end_line=2),
        SourceLocation(file="file1.py", start_line=5, end_line=8),
    ]
    assumption_entry = AssumptionEntry(text="Assumption 1", sources=source_location)
    column_constraints = ColumnConstraintsWithSources(code=code_entries, assumptions=[assumption_entry])
    constraints = ConstraintsWithSources(constraints={"column1": column_constraints})
    return constraints


@pytest.fixture
def gx_expectation_path():
    return get_project_root() / "tadv" / "llm" / "langchain" / "prompts" / "gx" / "expectations"
