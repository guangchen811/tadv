# https://github.com/schelterlabs/jenga/blob/master/src/jenga/basis.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class DataCorruption(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def transform(self, data):
        pass


class TabularCorruption(DataCorruption):
    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.fraction = 1.0

    @abstractmethod
    def transform(self, dataframe):
        pass

    @abstractmethod
    def identify_columns(self, dataframe):
        pass

    @staticmethod
    def validate_data(dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

    def sample_rows(self, data):
        if self.fraction == 1.0:
            rows = data.index
        # Completely At Random
        elif self.sampling.endswith('CAR'):
            rows = np.random.permutation(data.index)[:int(len(data) * self.fraction)]
        elif self.sampling.endswith('NAR') or self.sampling.endswith('AR'):
            n_values_to_discard = int(len(data) * min(self.fraction, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

            # Not At Random
            if self.sampling.endswith('NAR'):
                # pick a random percentile of values in this column
                rows = data[self.column].sort_values().iloc[perc_idx].index

            # At Random
            elif self.sampling.endswith('AR'):
                depends_on_col = np.random.choice(list(set(data.columns) - {self.column}))
                # pick a random percentile of values in other column
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows
