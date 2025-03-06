import oyaml as yaml

from tadv.data_models import CodeEntry, ColumnConstraints, \
    Constraints
from tadv.utils import get_project_root


def test_from_yaml(constraints_instance, tmp_path):
    constraints_instance.save_to_yaml(str(tmp_path / "constraints.yaml"))
    constraints = Constraints.from_yaml(str(tmp_path / "constraints.yaml"))

    assert constraints.to_dict() == constraints_instance.to_dict()


def test_from_dict():
    # Arrange: Define the input data structure
    data = {
        "constraints": {
            "column1": {
                "code": [
                    [".hasCompleteness('column1', lambda x: x > 0.9)", "Valid"],
                    [".hasUniqueness('column1')", "Invalid"]
                ],
                "assumptions": ["Assumption 1"]
            },
            "column2": {
                "code": [
                    [".hasCompleteness('column2', lambda x: x > 0.8)", "Valid"]
                ],
                "assumptions": ["Assumption 2"]
            }
        }
    }

    # Act: Create a Constraints instance using from_dict
    constraints = Constraints.from_dict(data)

    # Assert: Verify the top-level structure
    assert isinstance(constraints, Constraints)
    assert len(constraints.constraints) == 2

    # Assert: Verify column1
    column1 = constraints.constraints["column1"]
    assert isinstance(column1, ColumnConstraints)
    assert len(column1.code) == 2
    assert column1.code[0].suggestion == ".hasCompleteness('column1', lambda x: x > 0.9)"
    assert column1.code[0].validity == "Valid"
    assert column1.code[1].suggestion == ".hasUniqueness('column1')"
    assert column1.code[1].validity == "Invalid"
    assert column1.assumptions == ["Assumption 1"]

    # Assert: Verify column2
    column2 = constraints.constraints["column2"]
    assert isinstance(column2, ColumnConstraints)
    assert len(column2.code) == 1
    assert column2.code[0].suggestion == ".hasCompleteness('column2', lambda x: x > 0.8)"
    assert column2.code[0].validity == "Valid"
    assert column2.assumptions == ["Assumption 2"]


def test_to_dict(constraints_instance):
    expected_dict = {
        "constraints": {
            "column1": {
                "code": [
                    ["Ensure unique values", "Invalid"],
                    ["Use a non-null constraint", "Valid"]
                ],
                "assumptions": ["Assumption 1", "Assumption 2"]
            }
        }
    }
    assert constraints_instance.to_dict() == expected_dict


def test_to_string():
    # Arrange: Create a Constraints object with mock data
    constraints_data = {
        "column1": ColumnConstraints(
            code=[
                CodeEntry(suggestion=".hasCompleteness('column1', lambda x: x > 0.9)", validity="Valid"),
                CodeEntry(suggestion=".hasUniqueness('column1')", validity="Invalid"),
            ],
            assumptions=["Assumption 1"]
        ),
        "column2": ColumnConstraints(
            code=[
                CodeEntry(suggestion=".hasCompleteness('column2', lambda x: x > 0.8)", validity="Valid"),
            ],
            assumptions=["Assumption 2"]
        )
    }
    constraints = Constraints(constraints=constraints_data)

    # Act: Convert the constraints to a string
    result = constraints.to_string()

    # Assert: Verify the output is a valid YAML string
    expected_yaml = ('constraints:\n'
                     '  column1:\n'
                     '    code:\n'
                     "    - - '.hasCompleteness(''column1'', lambda x: x > 0.9)'\n"
                     '      - Valid\n'
                     "    - - .hasUniqueness('column1')\n"
                     '      - Invalid\n'
                     '    assumptions:\n'
                     '    - Assumption 1\n'
                     '  column2:\n'
                     '    code:\n'
                     "    - - '.hasCompleteness(''column2'', lambda x: x > 0.8)'\n"
                     '      - Valid\n'
                     '    assumptions:\n'
                     '    - Assumption 2\n')
    assert result == expected_yaml


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
    constraints._load_from_yaml(str(input_path))

    assert "column1" in constraints.constraints
    constraint = constraints.constraints["column1"]
    assert len(constraint.code) == 2
    assert constraint.code[0].suggestion == "Use a non-null constraint"
    assert constraint.code[0].validity == "Valid"
    assert constraint.assumptions == ["Assumption 1", "Assumption 2"]


def test_load_from_local_yaml():
    project_root = get_project_root()
    constraints = Constraints()
    constraints._load_from_yaml(f"{project_root}/tests/resources/constraints/example_cadv_constraints.yaml")

    assert "person_home_ownership" in constraints.constraints
    assert constraints.constraints["person_home_ownership"].code[0].validity == "Valid"


def test_from_llm_output():
    relevant_columns_list = ["column1", "column2"]
    suggestions = {
        "column1": [".hasCompleteness('column1', lambda x: x > 0.9)"],
        "column2": [".hasMax('column2', lambda x: x < 100)"],
        "column3": [".hasMin('column3', lambda x: x > 10)"]  # Not in relevant_columns_list
    }
    code_list_for_constraints_valid = [".hasCompleteness('column1', lambda x: x > 0.9)"]
    expectations = {
        "column1": ["Assumption 1"],
        "column2": ["Assumption 2"],
        "column3": ["Assumption 3"]  # Not in relevant_columns_list
    }

    constraints = Constraints.from_llm_output(relevant_columns_list, expectations, suggestions,
                                              code_list_for_constraints_valid)

    # Assert column1
    assert "column1" in constraints.constraints
    assert constraints.constraints["column1"].code[0].suggestion == ".hasCompleteness('column1', lambda x: x > 0.9)"
    assert constraints.constraints["column1"].code[0].validity == "Valid"
    assert constraints.constraints["column1"].assumptions == ["Assumption 1"]

    # Assert column2
    assert "column2" in constraints.constraints
    assert constraints.constraints["column2"].code[0].suggestion == ".hasMax('column2', lambda x: x < 100)"
    assert constraints.constraints["column2"].code[0].validity == "Invalid"
    assert constraints.constraints["column2"].assumptions == ["Assumption 2"]

    # Assert column3 (should not be included)
    assert "column3" not in constraints.constraints


def test_from_deequ_output():
    # Arrange: Define the input data for the test
    suggestion = [
        {"column_name": "column1", "code_for_constraint": ".hasCompleteness('column1', lambda x: x > 0.9)"},
        {"column_name": "column1", "code_for_constraint": ".hasUniqueness('column1')"},
        {"column_name": "column2", "code_for_constraint": ".hasMax('column2', lambda x: x < 100)"},
        {"column_name": "column3", "code_for_constraint": ".hasMin('column3', lambda x: x > 10)"},
    ]
    code_list_for_constraints_valid = [
        ".hasCompleteness('column1', lambda x: x > 0.9)",
        ".hasMax('column2', lambda x: x < 100)"
    ]

    # Act: Call the method to create a Constraints object
    constraints = Constraints.from_deequ_output(suggestion, code_list_for_constraints_valid)

    # Assert: Verify the constraints for column1
    assert "column1" in constraints.constraints
    column1 = constraints.constraints["column1"]
    assert len(column1.code) == 2
    assert column1.code[0].suggestion == ".hasCompleteness('column1', lambda x: x > 0.9)"
    assert column1.code[0].validity == "Valid"
    assert column1.code[1].suggestion == ".hasUniqueness('column1')"
    assert column1.code[1].validity == "Invalid"

    # Assert: Verify the constraints for column2
    assert "column2" in constraints.constraints
    column2 = constraints.constraints["column2"]
    assert len(column2.code) == 1
    assert column2.code[0].suggestion == ".hasMax('column2', lambda x: x < 100)"
    assert column2.code[0].validity == "Valid"

    # Assert: Verify the constraints for column3
    assert "column3" in constraints.constraints
    column3 = constraints.constraints["column3"]
    assert len(column3.code) == 1
    assert column3.code[0].suggestion == ".hasMin('column3', lambda x: x > 10)"
    assert column3.code[0].validity == "Invalid"


def test_get_suggestions_code_column_map():
    # Arrange: Mock constraints data
    constraints_data = {
        "column1": ColumnConstraints(
            code=[
                CodeEntry(suggestion=".hasCompleteness('column1', lambda x: x > 0.9)", validity="Valid"),
                CodeEntry(suggestion=".hasUniqueness('column1')", validity="Invalid"),
            ],
            assumptions=["Assumption 1"]
        ),
        "column2": ColumnConstraints(
            code=[
                CodeEntry(suggestion=".hasMax('column2', lambda x: x < 100)", validity="Valid"),
            ],
            assumptions=["Assumption 2"]
        ),
        "column3": ColumnConstraints(
            code=[
                CodeEntry(suggestion=".hasMin('column3', lambda x: x > 10)", validity="Invalid"),
            ],
            assumptions=[]
        )
    }
    constraints = Constraints(constraints=constraints_data)

    # Act: Get the suggestions-code-column map
    result_all = constraints.get_suggestions_code_column_map(valid_only=False)
    result_valid_only = constraints.get_suggestions_code_column_map(valid_only=True)

    # Assert: Verify the result when valid_only=False
    assert result_all == {
        ".hasCompleteness('column1', lambda x: x > 0.9)": "column1",
        ".hasUniqueness('column1')": "column1",
        ".hasMax('column2', lambda x: x < 100)": "column2",
        ".hasMin('column3', lambda x: x > 10)": "column3",
    }

    # Assert: Verify the result when valid_only=True
    assert result_valid_only == {
        ".hasCompleteness('column1', lambda x: x > 0.9)": "column1",
        ".hasMax('column2', lambda x: x < 100)": "column2",
    }
