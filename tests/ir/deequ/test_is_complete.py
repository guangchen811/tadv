import pytest

from tadv.ir_translator.deequ_constraints.base_constraint import DeequConstraint


def test_valid_constraint_with_hint():
    s = "isComplete(column='age', hint='no nulls')"
    c = DeequConstraint.from_string(s)
    assert c.constraint_type == "isComplete"
    assert c.params["column"] == "age"
    assert c.hint == "no nulls"
    assert c.to_deequ_code() == ".isComplete(column='age', hint='no nulls')"


def test_valid_constraint_with_hint_1():
    s = "isComplete(column=\"age\", hint='no nulls')"
    c = DeequConstraint.from_string(s)
    assert c.constraint_type == "isComplete"
    assert c.params["column"] == "age"
    assert c.hint == "no nulls"
    assert c.to_deequ_code() == ".isComplete(column='age', hint='no nulls')"


def test_valid_constraint_without_hint():
    s = "isComplete(column='age')"
    c = DeequConstraint.from_string(s)
    assert c.constraint_type == "isComplete"
    assert c.params["column"] == "age"
    assert c.hint is None
    assert c.to_deequ_code() == ".isComplete(column='age')"


def test_missing_column_raises():
    s = "isComplete(hint='no nulls')"
    with pytest.raises(ValueError, match="Column must be specified"):
        DeequConstraint.from_string(s)


def test_invalid_format_raises():
    s = "invalid_call"
    with pytest.raises(ValueError, match="Invalid input string"):
        DeequConstraint.from_string(s)
