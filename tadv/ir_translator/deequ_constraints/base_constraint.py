from abc import ABC, abstractmethod


class DeequConstraint(ABC):
    @abstractmethod
    def render(self) -> str:
        pass
