from typing import Dict, Any

from tadv.ir_translator.deequ_constraints.base_constraint import DeequConstraint


class GXExpectation():
    def __init__(self, expectation_type: str, kwargs: Dict[str, Any]):
        self.expectation_type = expectation_type
        self.kwargs = kwargs

    def to_deequ_constraint(self) -> 'DeequConstraint':
        raise NotImplementedError(
            f"to_deequ_constraint not implemented for {self.__class__.__name__}. "
            "Please implement this method in the subclass."
        )
