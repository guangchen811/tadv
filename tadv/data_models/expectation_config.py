from dataclasses import dataclass, field
from typing import List, Optional, Dict

import oyaml as yaml
import pandas as pd


@dataclass
class GXParameter:
    type: str
    description: str


@dataclass
class ExpectationConfig:
    Name: str
    URL: str
    Description: str
    Args: Dict[str, GXParameter]
    SampleData: pd.DataFrame
    Examples: Dict[str, str]
    Other_Parameters: Dict[str, GXParameter]
    Data_quality_issues: Optional[str] = None
    Related_Expectations: Optional[List[str]] = field(default_factory=list)

    @staticmethod
    def from_yaml_file(filepath: str):
        def parse_parameter_dict(d):
            return {k: GXParameter(**v) for k, v in d.items()}

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        example_data = data["Examples"]["Sample data"]
        example_df = pd.DataFrame({ex["column"]: ex["value_set"] for ex in example_data})
        return ExpectationConfig(
            Name=data.get("Name"),
            URL=data.get("URL"),
            Description=data.get("Description"),
            Args=parse_parameter_dict({k: v for param in data["Args"] for k, v in param.items()}),
            SampleData=example_df,
            Examples={
                "Passing case": data["Examples"].get("Passing case", []),
                "Failing case": data["Examples"].get("Failing case", [])
            },
            Other_Parameters=parse_parameter_dict(
                {k: v for param in data["Other Parameters"] for k, v in param.items()}),
            Data_quality_issues=data.get("Data quality issues"),
            Related_Expectations=data.get("Related Expectations", [])
        )
