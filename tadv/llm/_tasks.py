from enum import Enum, auto


class DVTask(Enum):
    EXPECTATION_EXTRACTION = auto()
    RELEVANT_COLUMN_TARGET = auto()
    RULE_GENERATION = auto()
