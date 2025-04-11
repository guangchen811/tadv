import pandas as pd

from tadv.dq_manager.gx.wrapper import GreatExpectationQualityManager


def test_gx_quality_manager():
    dq_manager = GreatExpectationQualityManager()
    df = pd.DataFrame({"a": ["foo", "bar", "baz"], "b": [1, 2, 3], "c": [5, 6, None]})

    dq_manager.apply_checks_from_strings(df, ["expect_column_values_to_be_in_set"])
