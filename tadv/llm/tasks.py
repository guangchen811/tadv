from enum import Enum, auto


class SequentialTADVTasks(Enum):
    """
    Enumeration of tasks for the Sequential TADV strategy.

    This strategy executes tasks in a fixed, linear order:
    1. COLUMN_ACCESS_DETECTION: Identify columns accessed by the downstream task.
    2. ASSUMPTION_EXTRACTION: Extract assumptions or expectations based on accessed columns.
    3. CODE_GENERATION: Generate constraint code from the extracted assumptions.
    """
    COLUMN_ACCESS_DETECTION = auto()
    ASSUMPTION_EXTRACTION = auto()
    CODE_GENERATION = auto()


class TreeStructuredTADVTasks(Enum):
    """
    Enumeration of tasks for the Tree-Structured TADV strategy.

    Unlike the sequential approach, this strategy uses a branching, hierarchical
    structure where different types of constraints (e.g., type, completeness, uniqueness,
    correlation) are extracted in parallel or layered flows.
    """
    pass
