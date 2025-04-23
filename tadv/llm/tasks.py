from enum import Enum, auto


class SequentialTADVTasks(Enum):
    COLUMN_ACCESS_DETECTION = auto()
    EXPECTATION_EXTRACTION = auto()
    CODE_GENERATION = auto()
