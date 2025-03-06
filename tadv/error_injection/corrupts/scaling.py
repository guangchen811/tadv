import numpy as np
import pandas as pd

from tadv.error_injection.abstract_corruption import TabularCorruption


class Scaling(TabularCorruption):

    def identify_columns(self, dataframe):
        raise NotImplementedError("Scaling corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])

        rows = self.sample_rows(dataframe)
        df.loc[rows, self.columns] *= scale_factor

        return df
