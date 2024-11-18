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
