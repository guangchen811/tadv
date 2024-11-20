import pytest
import oyaml as yaml

from cadv_exploration.data_models import CodeEntry, ColumnConstraints, \
    Constraints
from cadv_exploration.utils import get_project_root


@pytest.fixture
def constraints_instance():
    code_entries = [
        CodeEntry(suggestion="Use a non-null constraint", validity="Valid"),
        CodeEntry(suggestion="Ensure unique values", validity="Invalid"),
    ]
    column_constraints = ColumnConstraints(code=code_entries, assumptions=["Assumption 1", "Assumption 2"])
    constraints = Constraints(constraints={"column1": column_constraints})
    return constraints


def test_to_dict(constraints_instance):
    expected_dict = {
        "constraints": {
            "column1": {
                "code": [
                    ["Use a non-null constraint", "Valid"],
                    ["Ensure unique values", "Invalid"]
                ],
                "assumptions": ["Assumption 1", "Assumption 2"]
            }
        }
    }
    assert constraints_instance.to_dict() == expected_dict


def test_save_to_yaml(constraints_instance, tmp_path):
    output_path = tmp_path / "constraints.yaml"
    constraints_instance.save_to_yaml(str(output_path))

    with open(output_path, "r") as f:
        saved_data = yaml.safe_load(f)

    expected_data = constraints_instance.to_dict()
    assert saved_data == expected_data


def test_load_from_yaml(tmp_path):
    data = {
        "constraints": {
            "column1": {
                "code": [
                    ["Use a non-null constraint", "Valid"],
                    ["Ensure unique values", "Invalid"]
                ],
                "assumptions": ["Assumption 1", "Assumption 2"]
            }
        }
    }

    input_path = tmp_path / "constraints.yaml"
    with open(input_path, "w") as f:
        yaml.dump(data, f)

    constraints = Constraints()
    constraints.load_from_yaml(str(input_path))

    assert "column1" in constraints.constraints
    constraint = constraints.constraints["column1"]
    assert len(constraint.code) == 2
    assert constraint.code[0].suggestion == "Use a non-null constraint"
    assert constraint.code[0].validity == "Valid"
    assert constraint.assumptions == ["Assumption 1", "Assumption 2"]


def test_load_from_local_yaml():
    project_root = get_project_root()
    constraints = Constraints()
    constraints.load_from_yaml(f"{project_root}/tests/resources/constraints/example_cadv_constraints.yaml")

    assert "person_home_ownership" in constraints.constraints
    assert constraints.constraints["person_home_ownership"].code[0].validity == "Valid"


def test_from_llm_output():
    relevant_columns_list = ["column1", "column2"]
    suggestions = {
        "column1": [".hasCompleteness('column1', lambda x: x > 0.9)"],
        "column2": [".hasMax('column2', lambda x: x < 100)"]
    }
    code_list_for_constraints_valid = [".hasCompleteness('column1', lambda x: x > 0.9)"]
    expectations = {
        "column1": ["Assumption 1"],
        "column2": ["Assumption 2"]
    }

    constraints = Constraints.from_llm_output(relevant_columns_list, expectations, suggestions,
                                              code_list_for_constraints_valid)

    assert "column1" in constraints.constraints
    assert constraints.constraints["column1"].code[0].suggestion == ".hasCompleteness('column1', lambda x: x > 0.9)"
    assert constraints.constraints["column1"].code[0].validity == "Valid"
    assert constraints.constraints["column1"].assumptions == ["Assumption 1"]


def test_from_dict():
    data = {
        "constraints": {
            "column1": {
                "code": [
                    [".hasCompleteness('column1', lambda x: x > 0.9)", "Valid"]
                ],
                "assumptions": ["Assumption 1"]
            }
        }
    }

    constraints = Constraints.from_dict(data)

    assert "column1" in constraints.constraints
    assert constraints.constraints["column1"].code[0].suggestion == ".hasCompleteness('column1', lambda x: x > 0.9)"
    assert constraints.constraints["column1"].code[0].validity == "Valid"
    assert constraints.constraints["column1"].assumptions == ["Assumption 1"]
