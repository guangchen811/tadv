# https://github.com/schelterlabs/jenga/blob/master/src/jenga/basis.py
from abc import ABC, abstractmethod

import numpy as np


class DataCorruption(ABC):

    # Abstract base method for corruptions, they have to return a corrupted copied of the dataframe
    @abstractmethod
    def transform(self, data):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


class TabularCorruption(DataCorruption, ABC):
    """
    Corruptions for structured data
    Input:
    column: column to corrupt
    fraction: fraction of the column to corrupt
    sampling: sampling mechanism for corruptions, options are completely at random ('CAR'),at random ('AR'), not at random ('NAR')
    """

    def __init__(self, column, fraction, sampling):
        self.column = column
        self.fraction = fraction
        self.sampling = sampling

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


class SingleColumnTabularCorruption(TabularCorruption, ABC):
    """
    Corruptions for structured data that only affect a single column
    """
    pass


class MultiColumnTabularCorruption(TabularCorruption, ABC):
    """
    Corruptions for structured data that affect multiple columns
    """
    pass
