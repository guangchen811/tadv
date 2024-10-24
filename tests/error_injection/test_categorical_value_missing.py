import pandas as pd

from cadv_exploration.error_injection import MissingCategoricalValueCorruption
from cadv_exploration.utils import get_project_root, get_current_folder
from cadv_exploration.loader import load_csv


def test_on_small_dataset():
    df = pd.read_csv(get_current_folder() / "example_table.csv")

    missing_categorical_value_corruption_to_nan = MissingCategoricalValueCorruption(columns=["BloodType"],
                                                                                    severity=0.001,
                                                                                    corrupt_strategy="to_nan")
    corrupted_df_to_nan = missing_categorical_value_corruption_to_nan.corrupt(df)
    assert df.shape[0] == corrupted_df_to_nan.shape[0]
    assert corrupted_df_to_nan["BloodType"].isnull().sum() > 0
    assert len(corrupted_df_to_nan["BloodType"].dropna().unique()) < len(df["BloodType"].unique())

    missing_categorical_value_corruption_to_majority = MissingCategoricalValueCorruption(columns=["BloodType"],
                                                                                         severity=0.999,
                                                                                         corrupt_strategy="to_majority")
    corrupted_df_to_majority = missing_categorical_value_corruption_to_majority.corrupt(df)
    assert df.shape[0] == corrupted_df_to_majority.shape[0]
    assert corrupted_df_to_majority["BloodType"].isnull().sum() == 0
    assert len(corrupted_df_to_majority["BloodType"].unique()) == 1

    missing_categorical_value_corruption_to_random = MissingCategoricalValueCorruption(columns=["BloodType"],
                                                                                       severity=0.3,
                                                                                       corrupt_strategy="to_random")
    corrupted_df_to_random = missing_categorical_value_corruption_to_random.corrupt(df)
    assert corrupted_df_to_random["BloodType"].isnull().sum() == 0
    assert len(corrupted_df_to_random["BloodType"].unique()) < len(df["BloodType"].unique())

    assert df.shape[0] == corrupted_df_to_random.shape[0]
    missing_categorical_value_corruption_remove = MissingCategoricalValueCorruption(columns=["BloodType"], severity=0.3,
                                                                                    corrupt_strategy="remove")
    corrupted_df_remove = missing_categorical_value_corruption_remove.corrupt(df)
    assert df.shape[0] > corrupted_df_remove.shape[0]
    assert corrupted_df_remove["BloodType"].isnull().sum() == 0
    assert len(corrupted_df_remove["BloodType"].unique()) < len(df["BloodType"].unique())


def test_on_large_dataset():
    project_root = get_project_root()
    file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "files"
            / "healthcare_dataset.csv"
    )

    df = load_csv(file_path)
    missing_categorical_value_corruption_to_random = MissingCategoricalValueCorruption(columns=["Blood Type"],
                                                                                       severity=0.001,
                                                                                       corrupt_strategy="to_random")
    corrupted_df_to_random = missing_categorical_value_corruption_to_random.corrupt(df)
    assert df.shape[0] == corrupted_df_to_random.shape[0]
    assert corrupted_df_to_random["Blood Type"].isnull().sum() == 0
    assert len(corrupted_df_to_random["Blood Type"].unique()) < len(df["Blood Type"].unique())

    missing_categorical_value_corruption_to_majority = MissingCategoricalValueCorruption(columns=["Blood Type"],
                                                                                         severity=0.999,
                                                                                         corrupt_strategy="to_majority")
    corrupted_df_to_majority = missing_categorical_value_corruption_to_majority.corrupt(df)
    assert df.shape[0] == corrupted_df_to_majority.shape[0]
    assert corrupted_df_to_majority["Blood Type"].isnull().sum() == 0
    assert len(corrupted_df_to_majority["Blood Type"].unique()) == 1
