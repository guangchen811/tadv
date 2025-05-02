from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

import oyaml as yaml

from tadv.data_models.constraints import CodeEntry


@dataclass
class SourceLocation:
    file: str
    start_line: int
    end_line: int


@dataclass
class AssumptionEntry:
    text: str
    sources: List[SourceLocation] = field(default_factory=list)


@dataclass
class ColumnConstraintsWithSources:
    code: List[CodeEntry] = field(default_factory=list)
    assumptions: List[AssumptionEntry] = field(default_factory=list)


@dataclass
class ConstraintsWithSources:
    constraints: Dict[str, ColumnConstraintsWithSources] = field(default_factory=dict)

    def to_dict(self):
        # Convert the dataclass structure to a dictionary that yaml.dump can use
        return {
            "constraints": {
                column: {
                    "code": sorted([[entry.suggestion, entry.validity] for entry in constraint.code],
                                   key=lambda x: x[0]),
                    "assumptions": [
                        {"text": assumption.text, "sources": [source.__dict__ for source in assumption.sources]}
                        for assumption in constraint.assumptions
                    ]
                } for column, constraint in sorted(self.constraints.items())
            }
        }

    @classmethod
    def from_dict(cls, data: dict):
        constraints = cls()
        for column, constraint_data in data["constraints"].items():
            code_entries = [CodeEntry(suggestion=entry[0], validity=entry[1]) for entry in constraint_data["code"]]
            assumptions = [
                AssumptionEntry(text=assumption["text"],
                                sources=[SourceLocation(**source) for source in assumption["sources"]])
                for assumption in constraint_data["assumptions"]
            ]
            constraints.constraints[column] = ColumnConstraintsWithSources(code=code_entries, assumptions=assumptions)
        return constraints

    @classmethod
    def from_yaml(cls, input_path: Union[str, Path]):
        constraints = cls()
        constraints._load_from_yaml(input_path)
        return constraints

    def _load_from_yaml(self, input_path: Union[str, Path]):
        with open(input_path, "r") as f:
            data = yaml.safe_load(f)
            for column, constraint_data in data["constraints"].items():
                code_entries = [CodeEntry(suggestion=entry[0], validity=entry[1]) for entry in constraint_data["code"]]
                assumptions = [
                    AssumptionEntry(text=assumption["text"],
                                    sources=[SourceLocation(**source) for source in assumption["sources"]])
                    for assumption in constraint_data["assumptions"]
                ]
                self.constraints[column] = ColumnConstraintsWithSources(code=code_entries, assumptions=assumptions)

    def save_to_yaml(self, output_path: str):
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_llm_output(cls, accessed_columns_list, expectations, suggestions, code_list_for_constraints_valid):
        yaml_dict = {"constraints": {f"{relevant_column}": {"code": [], "assumptions": []} for relevant_column in
                                     accessed_columns_list}}
        for suggested_column, suggestions in suggestions.items():
            if suggested_column not in accessed_columns_list:
                continue
            for suggestion in suggestions:
                if suggestion in code_list_for_constraints_valid:
                    yaml_dict["constraints"][suggested_column]["code"].append([suggestion, "Valid"])
                else:
                    yaml_dict["constraints"][suggested_column]["code"].append([suggestion, "Invalid"])
        for suggested_column, expectations in expectations.items():
            if suggested_column not in accessed_columns_list:
                continue
            for expectation in expectations:
                yaml_dict["constraints"][suggested_column]["assumptions"].append(
                    {"text": expectation["text"], "sources": expectation["sources"]})
        return cls.from_dict(yaml_dict)

    def to_string(self):
        return yaml.dump(self.to_dict())
