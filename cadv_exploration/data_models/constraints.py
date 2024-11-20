from dataclasses import dataclass, field
from typing import List, Dict

import oyaml as yaml


@dataclass
class CodeEntry:
    suggestion: str
    validity: str  # "Valid" or "Invalid"


@dataclass
class ColumnConstraints:
    code: List[CodeEntry] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


@dataclass
class Constraints:
    constraints: Dict[str, ColumnConstraints] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, input_path: str):
        constraints = cls()
        constraints.load_from_yaml(input_path)
        return constraints

    def to_dict(self):
        # Convert the dataclass structure to a dictionary that yaml.dump can use
        return {
            "constraints": {
                column: {
                    "code": [[entry.suggestion, entry.validity] for entry in constraint.code],
                    "assumptions": constraint.assumptions
                } for column, constraint in self.constraints.items()
            }
        }

    @classmethod
    def from_llm_output(cls, relevant_columns_list, expectations, suggestions, code_list_for_constraints_valid):
        yaml_dict = {"constraints": {f"{relevant_column}": {"code": [], "assumptions": []} for relevant_column in
                                     relevant_columns_list}}
        for suggested_column, suggestions in suggestions.items():
            if suggested_column not in relevant_columns_list:
                continue
            for suggestion in suggestions:
                if suggestion in code_list_for_constraints_valid:
                    yaml_dict["constraints"][suggested_column]["code"].append([suggestion, "Valid"])
                else:
                    yaml_dict["constraints"][suggested_column]["code"].append([suggestion, "Invalid"])
        for suggested_column, expectations in expectations.items():
            if suggested_column not in relevant_columns_list:
                continue
            for expectation in expectations:
                yaml_dict["constraints"][suggested_column]["assumptions"].append(expectation)
        return cls.from_dict(yaml_dict)

    @classmethod
    def from_dict(cls, data):
        constraints = cls()
        for column, constraint in data["constraints"].items():
            constraints.constraints[column] = ColumnConstraints(
                code=[CodeEntry(suggestion=suggestion, validity=validity) for suggestion, validity in
                      constraint["code"]],
                assumptions=constraint["assumptions"]
            )
        return constraints

    def save_to_yaml(self, output_path: str):
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def load_from_yaml(self, input_path: str):
        with open(input_path, "r") as f:
            data = yaml.safe_load(f)
            for column, constraint in data["constraints"].items():
                self.constraints[column] = ColumnConstraints(
                    code=[CodeEntry(suggestion=suggestion, validity=validity) for suggestion, validity in
                          constraint["code"]],
                    assumptions=constraint["assumptions"]
                )

    def get_suggestions_code_column_map(self, valid_only=False):
        return {
            code.suggestion: column for column, constraint in self.constraints.items() for code in constraint.code if
            not valid_only or code.validity == "Valid"
        }
