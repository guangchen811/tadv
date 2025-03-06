import pandas as pd

from tadv.error_injection.corrupts import MaskValues
from tadv.utils import get_current_folder


def test_mask_values():
    df = pd.read_csv(get_current_folder() / "example_table.csv")

    mask_values = MaskValues(columns=["Age"], severity=0.3)

    corrupted_df = mask_values.transform(df)

    assert df.shape[0] == corrupted_df.shape[0]
    assert corrupted_df["Age"].isnull().sum() == 0
    assert len(corrupted_df["Age"].unique()) < len(df["Age"].unique())
    missing_value_count = corrupted_df["Age"].value_counts().get("missing", 0)
    na_value_count = corrupted_df["Age"].value_counts().get("NA", 0)
    question_mark_value_count = corrupted_df["Age"].value_counts().get("?", 0)
    assert missing_value_count > 0 or na_value_count > 0 or question_mark_value_count > 0
