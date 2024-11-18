from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ValidationResult:
    constraint: Constraint  # Reference to the validated constraint
    status: str  # e.g., "Passed" or "Failed"
    details: Dict[str, any] = field(default_factory=dict)  # Optional details like violations

