from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

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
    def from_yaml(cls, input_path: Union[str, Path]):
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
                    for entry in result["code"]
                ],
            )
        return results

    def check_result(self, column_skipped=None) -> tuple[int, int]:
        num_passed = 0
        num_failed = 0

        column_skipped = [] if column_skipped is None else column_skipped
        
        for column_name, column_result in self.results.items():
            if column_name in column_skipped:
                continue
            for entry in column_result.code:
                if entry.status == "Passed":
                    num_passed += 1
                elif entry.status == "Failed":
                    num_failed += 1

        return num_passed, num_failed
