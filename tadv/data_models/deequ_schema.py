from dataclasses import dataclass
from typing import Dict


@dataclass
class DeequParameter:
    description: str = ""
    type: str = None


@dataclass
class DeequSchema:
    Name: str
    Description: str
    RequiredArgs: Dict[str, DeequParameter]
    OptionalArgs: Dict[str, DeequParameter] = None

    @staticmethod
    def from_dict(data: Dict):
        assert len(data) == 1, "Data dictionary must contain exactly one key representing the function name."
        data_value = list(data.values())[0]
        required_args = {i: DeequParameter() for i in data_value['required']}
        optional_args = {i: DeequParameter() for i in data_value['optional']} if data_value['optional'] else {}

        return DeequSchema(
            Name=list(data.keys())[0],
            Description=data_value.get("description", ""),
            RequiredArgs=required_args,
            OptionalArgs=optional_args
        )
