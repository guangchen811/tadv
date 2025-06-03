from tadv.ir_translator.gx_expectations.base_expectation import GXExpectation


def test_expect_column_max_to_be_between():
    s = "ExpectColumnMaxToBeBetween(column='age', min_value=20, max_value=40, strict_min=False, strict_max=True)"
    expectation = GXExpectation.from_gx_code(s)

    assert expectation.expectation_type == "ExpectColumnMaxToBeBetween"
    assert expectation.kwargs["column"] == "age"
    assert expectation.kwargs["min_value"] == 20
    assert expectation.kwargs["max_value"] == 40
    assert expectation.kwargs["strict_min"] is False
    assert expectation.kwargs["strict_max"] is True

    deequ_constraint = expectation.to_deequ_constraint()

    assert deequ_constraint.constraint_type == "isContainedIn"
    assert deequ_constraint.params["column"] == "age"
    assert deequ_constraint.params["value_set"] == [20, 40]
    assert deequ_constraint.hint is None
