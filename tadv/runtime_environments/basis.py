from abc import abstractmethod, ABC
from pathlib import Path


class ExecutorBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, project_name: str, input_path: Path, script_path: Path, output_path: Path, timeout: int = 120):
        pass
