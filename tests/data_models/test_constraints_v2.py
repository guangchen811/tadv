from tadv.data_models import ConstraintsWithSources


def test_from_yaml(constraints_with_sources_instance, tmp_path):
    constraints_with_sources_instance.save_to_yaml(str(tmp_path / "constraints.yaml"))
    constraints = ConstraintsWithSources.from_yaml(str(tmp_path / "constraints.yaml"))

    assert constraints.to_dict() == constraints_with_sources_instance.to_dict()


def test_from_dict():
    data = {
        "constraints": {
            "column1": {
                "code": [["Code 1", "Valid"], ["Code 2", "Invalid"]],
                "assumptions": [
                    {"text": "Assumption 1", "sources": [{"file": "file1.py", "start_line": 1, "end_line": 2}]}
                ]
            },
            "column2": {
                "code": [["Use a unique constraint", "Valid"]],
                "assumptions": [
                    {"text": "Assumption 2", "sources": [{"file": "file2.py", "start_line": 3, "end_line": 4}]}
                ]
            }
        }
    }

    constraints = ConstraintsWithSources.from_dict(data)
    assert len(constraints.constraints) == 2
    assert constraints.constraints["column1"].code[0].suggestion == "Code 1"
    assert constraints.constraints["column1"].assumptions[0].text == "Assumption 1"
    assert constraints.constraints["column1"].assumptions[0].sources[0].file == "file1.py"
    assert constraints.constraints["column2"].code[0].suggestion == "Use a unique constraint"
    assert constraints.constraints["column2"].assumptions[0].text == "Assumption 2"
    assert constraints.constraints["column2"].assumptions[0].sources[0].file == "file2.py"


def test_to_dict(constraints_with_sources_instance):
    expected_dict = {
        "constraints": {
            "column1": {
                "code": [["Code 1", "Valid"], ["Code 2", "Invalid"]],
                "assumptions": [
                    {
                        "text": "Assumption 1",
                        "sources": [
                            {"file": "file1.py", "start_line": 1, "end_line": 2},
                            {"file": "file1.py", "start_line": 5, "end_line": 8}
                        ]
                    }
                ]
            }
        }
    }
    assert constraints_with_sources_instance.to_dict() == expected_dict


def test_to_string(constraints_with_sources_instance):
    expected_string = (
        "constraints:\n"
        "  column1:\n"
        "    code:\n"
        "    - - Code 1\n"
        "      - Valid\n"
        "    - - Code 2\n"
        "      - Invalid\n"
        "    assumptions:\n"
        "    - text: Assumption 1\n"
        "      sources:\n"
        "      - file: file1.py\n"
        "        start_line: 1\n"
        "        end_line: 2\n"
        "      - file: file1.py\n"
        "        start_line: 5\n"
        "        end_line: 8\n"
    )
    assert constraints_with_sources_instance.to_string() == expected_string


def test_save_to_yaml(constraints_with_sources_instance, tmp_path):
    output_path = tmp_path / "constraints.yaml"
    constraints_with_sources_instance.save_to_yaml(str(output_path))

    with open(output_path, "r") as f:
        content = f.read()

    assert "constraints:" in content
    assert "column1:" in content
    assert "Code 1" in content
    assert "Assumption 1" in content
    assert "file1.py" in content


def test_load_from_yaml(constraints_with_sources_instance, tmp_path):
    output_path = tmp_path / "constraints.yaml"
    constraints_with_sources_instance.save_to_yaml(str(output_path))

    loaded_constraints = ConstraintsWithSources.from_yaml(str(output_path))

    assert loaded_constraints.to_dict() == constraints_with_sources_instance.to_dict()


def test_from_llm_output():
    accessed_columns_list = ["column1", "column2"]
    expectations = {
        "column1": [{"text": "Assumption 1", "sources": [{"file": "file1.py", "start_line": 1, "end_line": 2}]}],
        "column2": [{"text": "Assumption 2", "sources": [{"file": "file2.py", "start_line": 3, "end_line": 4}]}]
    }
    suggestions = {
        "column1": ["Code 1", "Code 2"],
        "column2": ["Code 3"]
    }
    code_list_for_constraints_valid = ["Code 1", "Code 2"]

    constraints = ConstraintsWithSources.from_llm_output(accessed_columns_list, expectations, suggestions,
                                                         code_list_for_constraints_valid)
    assert len(constraints.constraints) == 2
    assert constraints.constraints["column1"].code[0].suggestion == "Code 1"
    assert constraints.constraints["column1"].assumptions[0].text == "Assumption 1"
    assert constraints.constraints["column1"].assumptions[0].sources[0].file == "file1.py"
    assert constraints.constraints["column2"].code[0].suggestion == "Code 3"
    assert constraints.constraints["column2"].assumptions[0].text == "Assumption 2"
    assert constraints.constraints["column2"].assumptions[0].sources[0].file == "file2.py"
