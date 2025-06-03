import ast
import re
from typing import Dict, Any

from tadv.ir_translator.deequ_constraints.base_constraint import DeequConstraint
from tadv.ir_translator.gx_expectations.function_manager import GXFunctionManager


class GXExpectation():
    def __init__(self, expectation_type: str, kwargs: Dict[str, Any]):
        self.expectation_type = expectation_type
        self.kwargs = kwargs

    @classmethod
    def from_gx_code(cls, expectation_string: str) -> 'GXExpectation':
        """
        Parses a Great Expectations expectation string and returns an instance of GXExpectation.
        Example input: "expect_column_values_to_be_in_set(column='age', value_set=[20, 30, 40])"
        """
        match = re.match(
            r"(\w+)\((.*)\)", expectation_string.strip()
        )
        if not match:
            raise ValueError(f"Invalid expectation string: {expectation_string}")
        expectation_type, args_str = match.groups()

        fake_call = f"f({args_str})"
        try:
            tree = ast.parse(fake_call, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Malformed argument string: {args_str}") from e
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Parsed expression is not a call")
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
        cls.validate_params(kwargs, expectation_type)
        return cls(expectation_type=expectation_type, kwargs=kwargs)

    @staticmethod
    def validate_params(kwargs: Dict[str, Any], expectation_type: str):
        """
        Validates the parameters for the given expectation type.
        This method should be implemented in subclasses to enforce specific validation rules.
        """
        GXFunctionManager().get_expectation(expectation_type)
        required_params = GXFunctionManager().get_expectation(expectation_type).Args.keys()
        optional_params = GXFunctionManager().get_expectation(expectation_type).Other_Parameters.keys()
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter '{param}' for expectation '{expectation_type}'")
        for param in kwargs:
            if param not in required_params and param not in optional_params:
                raise ValueError(f"Unexpected parameter '{param}' for expectation '{expectation_type}'")

    def to_deequ_constraint(self) -> 'DeequConstraint':
        raise NotImplementedError(
            f"to_deequ_constraint not implemented for {self.__class__.__name__}. "
            "Please implement this method in the subclass."
        )
