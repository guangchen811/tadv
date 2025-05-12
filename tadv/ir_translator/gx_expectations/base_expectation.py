from abc import abstractmethod, ABC
from typing import Dict, Any

from tadv.ir_translator.deequ_constraints.base_constraint import DeequConstraint


class GXExpectation(ABC):
    def __init__(self, expectation_type: str, kwargs: Dict[str, Any]):
        self.expectation_type = expectation_type
        self.kwargs = kwargs

    @abstractmethod
    def to_deequ_constraint(self) -> 'DeequConstraint':
        pass
