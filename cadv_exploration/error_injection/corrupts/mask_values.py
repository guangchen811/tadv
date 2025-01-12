import numpy as np
import pandas as pd

from cadv_exploration.error_injection.abstract_corruption import TabularCorruption


class MaskValues(TabularCorruption):

    def identify_columns(self, dataframe):
        raise NotImplementedError("MaskValues corruption does not support column identification yet.")

    def transform(self, dataframe: pd.DataFrame):
        df = dataframe.copy(deep=True)

        mask_value = np.random.choice(['?', 'NA', 'missing'])

        for col in self.columns:
            rows = self.sample_rows(dataframe)
            df.loc[rows, col] = mask_value
        return df
