from enum import Enum, auto


class DVTask(Enum):
    EXPECTATION_EXTRACTION = auto()
    COLUMN_ACCESS_DETECTION = auto()
    RULE_GENERATION = auto()
