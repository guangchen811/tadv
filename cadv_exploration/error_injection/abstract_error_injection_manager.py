from abc import ABC, abstractmethod

from cadv_exploration.error_injection.abstract_corruption import DataCorruption


class AbstractErrorInjectionManager(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def error_injection(self, corrupts: list[DataCorruption]):
        raise NotImplementedError

    def _create_processed_data_path(self, processed_data_dir):
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_idx = len(list(processed_data_dir.iterdir()))
        processed_data_path = processed_data_dir / f"{processed_data_idx}"
        processed_data_path.mkdir(parents=True, exist_ok=True)
        return processed_data_path
