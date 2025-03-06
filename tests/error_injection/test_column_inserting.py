import pandas as pd

from tadv.error_injection.corrupts import ColumnInserting
from tadv.utils import get_current_folder


def test_column_inserting():
    df = pd.read_csv(get_current_folder() / "example_table.csv")
    inserting_by_concat = ColumnInserting(columns=["FirstName", "LastName"], corrupt_strategy="concatenate")
    inserting_by_adding_prefix = ColumnInserting(columns=["FirstName", "UserName"], corrupt_strategy="add_prefix")
    inserting_by_sanitizing = ColumnInserting(columns="UserName", corrupt_strategy="sanitize_to_identifier")

    df_concat = inserting_by_concat.transform(df)
    df_adding_prefix = inserting_by_adding_prefix.transform(df)
    df_sanitizing = inserting_by_sanitizing.transform(df)

    assert df_concat.shape[1] == df.shape[1] + 1
    assert df_adding_prefix.shape[1] == df.shape[1] + 2
    assert df_sanitizing.shape[1] == df.shape[1] + 1

    assert df_concat.columns[-1] == "FirstName_LastName"
    assert df_adding_prefix.columns[-1] == "corrupted_UserName"
    assert df_sanitizing.columns[-1] == "UserName_sanitized"

    assert df_concat["FirstName_LastName"].equals(df["FirstName"] + "_" + df["LastName"])
    assert df_adding_prefix["corrupted_UserName"].equals("corrupted_" + df["UserName"])
    assert df_sanitizing["UserName_sanitized"].equals(df["UserName"].apply(ColumnInserting._sanitize_to_identifier))
