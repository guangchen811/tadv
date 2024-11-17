import numpy as np

from cadv_exploration.error_injection.basis import TabularCorruption


class Scaling(TabularCorruption):

    def identify_columns(self, dataframe):
        raise NotImplementedError("Scaling corruption does not require column identification.")

    def transform(self, dataframe: object) -> object:
        df = dataframe.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])

        rows = self.sample_rows(dataframe)
        df.loc[rows, self.columns] *= scale_factor

        return df
