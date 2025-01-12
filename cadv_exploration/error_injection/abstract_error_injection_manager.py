from abc import ABC, abstractmethod

from cadv_exploration.error_injection.abstract_corruption import DataCorruption


class AbstractErrorInjectionManager(ABC):
    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def error_injection(self, corrupts: list[DataCorruption]):
        raise NotImplementedError
