from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union

import yaml


@dataclass
class Example:
    column: str
    value_set: List[Union[str, int, float]]


@dataclass
class Parameter:
    type: str
    description: str


@dataclass
class ExpectationConfig:
    Name: str
    URL: str
    Description: str
    Args: Dict[str, Parameter]
    Examples: Dict[str, List[Example]]
    Other_Parameters: Dict[str, Parameter]
    Data_quality_issues: Optional[str] = None
    Related_Expectations: Optional[List[str]] = field(default_factory=list)

    @staticmethod
    def from_yaml_file(filepath: str) -> "ExpectationConfig":
        def parse_parameter_dict(d):
            return {k: Parameter(**v) for k, v in d.items()}

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        return ExpectationConfig(
            Name=data.get("Name"),
            URL=data.get("URL"),
            Description=data.get("Description"),
            Args=parse_parameter_dict({k: v for param in data["Args"] for k, v in param.items()}),
            Examples={
                "Sample data": [Example(**ex) for ex in data["Examples"]["Sample data"]],
                "Passing case": data["Examples"].get("Passing case", []),
                "Failing case": data["Examples"].get("Failing case", [])
            },
            Other_Parameters=parse_parameter_dict(
                {k: v for param in data["Other Parameters"] for k, v in param.items()}),
            Data_quality_issues=data.get("Data quality issues"),
            Related_Expectations=data.get("Related Expectations", [])
        )


if __name__ == "__main__":
    config = ExpectationConfig.from_yaml_file("expectations/expect_column_distinct_values_to_equal_set.yaml")
    print(config.Name)
    print(config.Args["column"].description)
    print(config.Examples["Sample data"][0].column)
    print(config.Examples["Sample data"][0].value_set)
    print(config.Examples["Passing case"])
    print(config.Examples["Failing case"])
    print(config.Other_Parameters)
    print(config.Data_quality_issues)
    print(config.Related_Expectations)
    print(config.URL)
    print(config.Description)
