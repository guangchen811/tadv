from enum import Enum, auto


class SequentialTADVTasks(Enum):
    """
    Enum for sequential TADV tasks.
    sequential TADV consists of three tasks:
    1. COLUMN_ACCESS_DETECTION: Detects the columns accessed by the downstream task.
    2. EXPECTATION_EXTRACTION: Extracts the expectations from the downstream task code on the columns accessed.
    3. CODE_GENERATION: Generates the constraint code based on the expectations extracted.
    """
    COLUMN_ACCESS_DETECTION = auto()
    EXPECTATION_EXTRACTION = auto()
    CODE_GENERATION = auto()
