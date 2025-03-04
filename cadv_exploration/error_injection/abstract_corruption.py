# https://github.com/schelterlabs/jenga/blob/master/src/jenga/basis.py
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DataCorruption(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass


class TabularCorruption(DataCorruption):
    def __init__(self, columns=None, severity=None, sampling=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.severity = 1.0 if severity is None else severity
        self.sampling = 'CAR' if sampling is None else sampling

    def to_dict(self):
        return {
            self.__class__.__name__: {
                "Columns": self.columns,
                "Params": {k: v for k, v in self.__dict__.items() if k != "columns"},
            }
        }

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame):
        pass

    @abstractmethod
    def identify_columns(self, dataframe):
        pass

    @staticmethod
    def validate_data(dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

    def sample_rows(self, data):

        if self.severity == 1.0:
            rows = data.index
        # Completely At Random
        elif self.sampling.endswith('CAR'):
            rows = np.random.permutation(data.index)[:int(len(data) * self.severity)]
        elif self.sampling.endswith('NAR') or self.sampling.endswith('AR'):
            n_values_to_discard = int(len(data) * min(self.severity, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)
            # Not At Random
            if self.sampling.endswith('NAR'):
                rows = data[self.columns].sort_values().iloc[perc_idx].index
            # At Random
            elif self.sampling.endswith('AR'):
                depends_on_col = np.random.choice(list(set(data.columns) - {self.columns}))
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            raise ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows
