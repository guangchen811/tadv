from typing import List

from tadv.ir_translator.deequ_constraints.base_constraint import DeequConstraint
from tadv.ir_translator.gx_expectations.base_expectation import GXExpectation
from tadv.ir_translator.translator.base_translator import ExpectationTranslator


class SimpleTranslator(ExpectationTranslator):
    """
    A simple translator that maps known Expectation types to DeequConstraints.
    """

    def translate(self, expectation: GXExpectation, to: str = "deequ") -> DeequConstraint:
        """
        Translate a single Expectation into a DeequConstraint.
        """
        # Here you would implement the logic to translate the expectation
        # For example:
        raise ValueError(f"Unknown expectation type: {expectation.expectation_type}")


def batch_translate(self, expectations: List[GXExpectation], to: str = "deequ") -> List[DeequConstraint]:
    """
    Translate a list of Expectations into a list of DeequConstraints.
    """
    constraints = []
    for exp in expectations:
        constraint = self.translate(exp)
        constraints.append(constraint)
    return constraints
