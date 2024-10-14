from enum import Enum, auto


class DVTask(Enum):
    EXPECTATION_EXTRACTION = auto()
    RELEVENT_COLUMN_TARGET = auto()
    CHECK_FORMULATION = auto()
