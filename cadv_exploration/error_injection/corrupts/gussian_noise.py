import random

import numpy as np
import pandas as pd
from numpy import dtype

from cadv_exploration.error_injection.abstract_corruption import TabularCorruption


class GaussianNoise(TabularCorruption):

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def identify_columns(self, dataframe):
        raise NotImplementedError("GaussianNoise corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)
        for col in self.columns:
            self._transform_column(df, col)
        return df

    def _transform_column(self, dataframe, column):
        stddev = np.std(dataframe[column])
        scale = random.uniform(1, 5)

        if self.severity > 0:
            rows = self.sample_rows(dataframe)
            noise = np.random.normal(0, scale * stddev, size=len(rows))
            if dataframe[column].dtype == dtype('int64'):
                noise = noise.astype(int)
            dataframe.loc[rows, column] += noise
        return dataframe
