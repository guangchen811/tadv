# https://github.com/schelterlabs/jenga/blob/master/src/jenga/basis.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class DataCorruption(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def corrupt(self, data):
        pass


class TabularCorruption(DataCorruption):
    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    @abstractmethod
    def corrupt(self, dataframe):
        pass

    @abstractmethod
    def identify_columns(self, dataframe):
        pass

    @staticmethod
    def validate_data(dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
