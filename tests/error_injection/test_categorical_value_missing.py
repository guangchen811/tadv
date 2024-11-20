import pandas as pd

from cadv_exploration.error_injection import MissingCategoricalValueCorruption
from cadv_exploration.utils import get_current_folder


def test_on_small_dataset():
    df = pd.read_csv(get_current_folder() / "example_table.csv")

    missing_categorical_value_corruption_to_nan = MissingCategoricalValueCorruption(columns=["BloodType"],
                                                                                    severity=0.001,
                                                                                    corrupt_strategy="to_nan")
    corrupted_df_to_nan = missing_categorical_value_corruption_to_nan.transform(df)
    assert df.shape[0] == corrupted_df_to_nan.shape[0]
    assert corrupted_df_to_nan["BloodType"].isnull().sum() > 0
    assert len(corrupted_df_to_nan["BloodType"].dropna().unique()) < len(df["BloodType"].unique())

    missing_categorical_value_corruption_to_majority = MissingCategoricalValueCorruption(columns=["BloodType"],
                                                                                         severity=0.999,
                                                                                         corrupt_strategy="to_majority")
    corrupted_df_to_majority = missing_categorical_value_corruption_to_majority.transform(df)
    assert df.shape[0] == corrupted_df_to_majority.shape[0]
    assert corrupted_df_to_majority["BloodType"].isnull().sum() == 0
    assert len(corrupted_df_to_majority["BloodType"].unique()) == 1

    missing_categorical_value_corruption_to_random = MissingCategoricalValueCorruption(columns=["BloodType"],
                                                                                       severity=0.3,
                                                                                       corrupt_strategy="to_random")
    corrupted_df_to_random = missing_categorical_value_corruption_to_random.transform(df)
    assert corrupted_df_to_random["BloodType"].isnull().sum() == 0
    assert len(corrupted_df_to_random["BloodType"].unique()) < len(df["BloodType"].unique())

    assert df.shape[0] == corrupted_df_to_random.shape[0]
    missing_categorical_value_corruption_remove = MissingCategoricalValueCorruption(columns=["BloodType"], severity=0.3,
                                                                                    corrupt_strategy="remove")
    corrupted_df_remove = missing_categorical_value_corruption_remove.transform(df)
    assert df.shape[0] > corrupted_df_remove.shape[0]
    assert corrupted_df_remove["BloodType"].isnull().sum() == 0
    assert len(corrupted_df_remove["BloodType"].unique()) < len(df["BloodType"].unique())
