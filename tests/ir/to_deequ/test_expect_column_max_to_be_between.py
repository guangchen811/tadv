from tadv.ir_translator.gx_expectations.base_expectation import GXExpectation


def test_expect_column_values_to_be_in_set():
    s = "ExpectColumnValuesToBeInSet(column='age', value_set=[20, 30, 40])"
    expectation = GXExpectation.from_gx_code(s)
    #
    deequ_constraint = expectation.to_deequ_constraint()
    assert deequ_constraint.constraint_type == "isContainedIn"
    assert deequ_constraint.params["column"] == "age"
    assert deequ_constraint.params["allowed_values"] == [20, 30, 40]
    assert deequ_constraint.hint is None
