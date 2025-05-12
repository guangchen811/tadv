# translator/base_translator.py

from abc import ABC, abstractmethod
from typing import List

from tadv.ir_translator.deequ_constraints.base_constraint import DeequConstraint
from tadv.ir_translator.gx_expectations.base_expectation import GXExpectation


class ExpectationTranslator(ABC):
    """
    Abstract base class for translating Expectations into DeequConstraints.
    """

    @abstractmethod
    def translate(self, expectation: GXExpectation, to: str = "deequ") -> DeequConstraint:
        """
        Translate a single GXExpectation into a DeequConstraint.
        """
        pass

    @abstractmethod
    def batch_translate(self, expectations: List[GXExpectation], to: str = "deequ") -> List[DeequConstraint]:
        """
        Translate a list of GXExpectations into a list of DeequConstraints.
        """
        pass
