from cadv_exploration.error_injection.abstract_corruption import TabularCorruption


class ColumnDropping(TabularCorruption):
    def __init__(self, columns, severity=1, **kwargs):
        super().__init__(columns, severity=severity, **kwargs)

    def identify_columns(self, dataframe):
        return self.columns

    def transform(self, dataframe, **kwargs):
        df = dataframe.copy(deep=True)
        df.drop(columns=self.columns, inplace=True)
        return df
