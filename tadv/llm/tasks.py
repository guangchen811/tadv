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

    Each enum member represents a node type in the branching constraint generation tree.
    The structure enables parallel and modular extraction of diverse constraint types
    across different semantic categories of data.
    """

    # Root level
    COLUMN_CATEGORIZATION = auto()

    # First-level branches (constraint categories)
    TYPE_CONSTRAINTS = auto()
    COMPLETENESS_CONSTRAINTS = auto()
    VALUE_DOMAIN_CONSTRAINTS = auto()
    UNIQUENESS_CONSTRAINTS = auto()
    REFERENTIAL_CONSTRAINTS = auto()
    DISTRIBUTIONAL_CONSTRAINTS = auto()
    DERIVED_COLUMN_CONSTRAINTS = auto()
    CONDITIONAL_CONSTRAINTS = auto()
    SEMANTIC_CODE_CONSTRAINTS = auto()

    # Leaf-level specialization (can be expanded as needed)
    NUMERIC_RANGE_CONSTRAINTS = auto()
    CATEGORICAL_MEMBERSHIP_CONSTRAINTS = auto()
    MISSING_VALUE_THRESHOLD_CONSTRAINTS = auto()
    DATE_PARSABILITY_CONSTRAINTS = auto()
    ASSERTION_INFERRED_CONSTRAINTS = auto()
    CROSS_COLUMN_DEPENDENCY_CONSTRAINTS = auto()

    CODE_GENERATION = auto()
