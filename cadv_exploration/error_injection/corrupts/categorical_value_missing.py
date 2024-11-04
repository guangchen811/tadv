import numpy as np

from cadv_exploration.error_injection.basis import TabularCorruption


class MissingCategoricalValueCorruption(TabularCorruption):
    def __init__(self, columns=None, severity=0.1, corrupt_strategy="to_nan", **kwargs):
        super().__init__(columns, **kwargs)
        self.severity = severity
        self.max_unique_num = 30
        self.corrupt_strategy = corrupt_strategy
        assert self.corrupt_strategy in ["to_nan", "to_majority", "to_random", "remove"]

    def identify_columns(self, dataframe):
        """
        Identify columns in the dataframe that likely have categorical data.
        This is based on the number of unique values in proportion to the total number of rows.
        """
        categorical_columns = []
        total_rows = len(dataframe)

        for col in dataframe.columns:
            # Check if the column is of object type or has relatively few unique values
            num_unique_values = dataframe[col].nunique()

            # Heuristic: consider the column categorical if it has less than max_unique_ratio of unique values
            if num_unique_values < self.max_unique_num:
                categorical_columns.append(col)

        return categorical_columns

    def transform(self, dataframe):
        self.validate_data(dataframe)
        if self.columns is None:
            self.columns = self.identify_columns(dataframe)
        corrupted_df = dataframe.copy()
        for col in self.columns:
            corrupted_df = self._corrupt_column(corrupted_df, col)
        return corrupted_df

    def _corrupt_column(self, dataframe, column):
        corrupted_df = dataframe.copy()

        value_counts = corrupted_df[column].value_counts()
        total_records_to_remove = int(self.severity * len(corrupted_df))
        sorted_categories = value_counts.sort_values(ascending=True).index

        categories_to_remove = []
        removed_records = 0
        for category in sorted_categories:
            category_count = value_counts[category]
            # make sure we don't remove more records than needed but also remove at least one category
            if (removed_records + category_count > total_records_to_remove) and (len(categories_to_remove) > 0):
                break
            categories_to_remove.append(category)
            removed_records += category_count

        if removed_records < total_records_to_remove:
            categories_to_remove.append(sorted_categories[len(categories_to_remove)])

        remove_mask = corrupted_df[column].isin(categories_to_remove)

        if self.corrupt_strategy == "to_nan":
            corrupted_df[column] = corrupted_df[column].astype("object")
            corrupted_df.loc[remove_mask, column] = np.nan
        elif self.corrupt_strategy == "to_majority":
            corrupted_df[column] = corrupted_df[column].astype("object")
            majority = corrupted_df[column].mode().values[0]
            corrupted_df.loc[remove_mask, column] = majority
        elif self.corrupt_strategy == "to_random":
            corrupted_df[column] = corrupted_df[column].astype("object")
            unique_values = set(corrupted_df[column].unique()) - set(categories_to_remove)
            unique_values = list(unique_values)
            corrupted_df.loc[remove_mask, column] = np.random.choice(unique_values, remove_mask.sum())
        elif self.corrupt_strategy == "remove":
            corrupted_df = corrupted_df[~remove_mask]

        # Calculate the percentage of data that was actually removed or altered
        actual_changed_percent = remove_mask.mean() * 100

        # If the severity percent was not fully reached, continue corrupting more rows using the strategy
        if actual_changed_percent < self.severity * 100:
            remaining_rows_to_corrupt = int((self.severity * len(corrupted_df)) - remove_mask.sum())
            if remaining_rows_to_corrupt > 0:
                additional_mask = np.random.rand(len(corrupted_df)) < (remaining_rows_to_corrupt / len(corrupted_df))

                # Apply corruption strategy on these additional rows
                if self.corrupt_strategy == "to_nan":
                    corrupted_df.loc[additional_mask, column] = np.nan
                elif self.corrupt_strategy == "to_majority":
                    corrupted_df.loc[additional_mask, column] = majority
                elif self.corrupt_strategy == "to_random":
                    corrupted_df.loc[additional_mask, column] = np.random.choice(unique_values, additional_mask.sum())

        return corrupted_df
