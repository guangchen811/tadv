from unittest.mock import patch

import pandas as pd

from tadv.data_models import Constraints


def test_get_constraints_for_spark_df(dq_manager):
    """
    Test the get_constraints_for_spark_df method to ensure it returns a Constraints object
    with valid, filtered constraints.
    """
    # Prepare test data
    df = pd.DataFrame({
        "a": ["apple", "banana", "cherry"],
        "b": [1, 2, 3],
        "c": [None, 1.0, 2.0]
    })

    # Convert to Spark DataFrame
    spark_df, spark = dq_manager.spark_df_from_pandas_df(df)

    # Run method under test
    constraints = dq_manager.get_constraints_for_spark_df(spark, spark_df)

    # Assertions
    assert isinstance(constraints, Constraints)

    # If Constraints class has a method like `.to_list()` or `.rules`, we can inspect content
    rule_list = constraints.rules if hasattr(constraints, "rules") else []
    assert isinstance(rule_list, list)

    # Optionally check that all constraints are strings or dicts
    for rule in rule_list:
        assert isinstance(rule, dict), f"Unexpected rule type: {type(rule)}"
        assert "code_for_constraint" in rule or "column" in rule, f"Unexpected rule keys: {rule}"
    spark.sparkContext._gateway.shutdown_callback_server()
    spark.stop()


def test_filter_constraints_only_success(dq_manager):
    code_list = [
        "assertColumnIsComplete('a')",
        "assertColumnHasDataType('b', 'Integral')",
        "invalidSyntaxHere("
    ]

    # Simulate validation results for each constraint
    fake_validation_results = [
        {'constraint_status': 'Success'},
        {'constraint_status': 'Failure'},
        None  # e.g., due to invalid syntax
    ]

    with patch.object(dq_manager, 'apply_checks_from_strings', return_value=fake_validation_results):
        filtered = dq_manager.filter_constraints(
            code_list_for_constraints=code_list,
            spark_original_validation=None,
            spark_original_validation_df=None
        )

    # Only the first one should survive
    assert filtered == ["assertColumnIsComplete('a')"]
