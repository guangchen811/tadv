from dataclasses import dataclass, field
from typing import List, Dict

import oyaml as yaml


@dataclass
class ValidationCodeEntry:
    suggestion: str
    status: str  # "Passed" or "Failed"


@dataclass
class ColumnValidationResults:
    code: List[ValidationCodeEntry] = field(default_factory=list)


@dataclass
class ValidationResults:
    results: Dict[str, ColumnValidationResults] = field(default_factory=dict)

    def to_dict(self):
        return {
            "results": {
                column: {
                    "code": [
                        [entry.suggestion, entry.status]
                        for entry in validation_result.code
                    ]
                }
                for column, validation_result in self.results.items()
            }
        }

    def save_to_yaml(self, output_path: str):
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_yaml(cls, input_path: str):
        with open(input_path, "r") as f:
            data = yaml.safe_load(f)
            results = cls()
            for column, code_entry in data["results"].items():
                results.results[column] = ColumnValidationResults(
                    code=[
                        ValidationCodeEntry(
                            suggestion=entry[0],
                            status=entry[1],
                        )
                        for entry in code_entry['code']
                    ]
                )
            return results

    @classmethod
    def from_dict(cls, data: dict):
        results = cls()
        for column, result in data["results"].items():
            results.results[column] = ColumnValidationResults(
                code=[
                    ValidationCodeEntry(
                        suggestion=entry[0],
                        status=entry[1],
                    )
                    for entry in result
                ],
            )
        return results
