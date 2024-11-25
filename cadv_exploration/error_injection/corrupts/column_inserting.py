import re
from typing import Union, List

import pandas as pd

from cadv_exploration.error_injection.basis import TabularCorruption


class ColumnInserting(TabularCorruption):
    def __init__(self, columns: Union[str, List[str]], severity: int = 1, corrupt_strategy: str = None, **kwargs):
        super().__init__(columns, **kwargs)
        self.severity = severity
        if corrupt_strategy is None:
            raise ValueError("Corrupt strategy must be specified.")
        corrupt_strategy_options = ["add_prefix", "concatenate", "sanitize_to_identifier"]
        if corrupt_strategy not in corrupt_strategy_options:
            raise ValueError(f"Corrupt strategy must be one of {corrupt_strategy_options}")
        self.corrupt_strategy = corrupt_strategy

    def identify_columns(self, dataframe):
        raise NotImplementedError("ColumnInserting corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame, **kwargs):
        df = dataframe.copy(deep=True)
        if self.corrupt_strategy == "concatenate":
            assert isinstance(self.columns, List) and len(self.columns) == 2
        if isinstance(self.columns, str):
            self.columns = [self.columns]
        for col in self.columns:
            df = self._transform_column(df, col)
        return df

    def _transform_column(self, dataframe, column):
        if self.corrupt_strategy == "add_prefix":
            prefix = "corrupted_"
            new_column = prefix + column
            dataframe[new_column] = dataframe[column].apply(lambda x: prefix + x)
        elif self.corrupt_strategy == "concatenate":
            new_column = "_".join(self.columns)
            dataframe[new_column] = dataframe[self.columns[0]] + "_" + dataframe[self.columns[1]]
        elif self.corrupt_strategy == "sanitize_to_identifier":
            new_column = column + "_sanitized"
            dataframe[new_column] = dataframe[column].apply(self._sanitize_to_identifier)
        return dataframe

    @staticmethod
    def _sanitize_to_identifier(name: str) -> str:
        identifier = re.sub(r'\W|^(?=\d)', '_', name)
        return identifier
