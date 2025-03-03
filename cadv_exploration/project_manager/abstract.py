from abc import ABC
from typing import List

from utils import get_project_root


class AbstractProjectManager(ABC):
    def __init__(self, dataset_name, downstream_task_types=None):
        super().__init__()
        self.project_root = get_project_root()
        self._available_datasets = [item.name for item in self.project_data_path.iterdir()]
        self.dataset_path = self.project_data_path / dataset_name
        if not self.dataset_path.exists():
            raise ValueError(
                f"Dataset {dataset_name} is not available in the project data path. Available datasets are: {self._available_datasets}")
        if self.scripts_path.exists():
            self._available_tasks = [item.name for item in self.scripts_path.iterdir()]
        else:
            self._available_tasks = []
        if downstream_task_types is not None and downstream_task_types not in self._available_tasks:
            raise ValueError(
                f"Downstream task type {downstream_task_types} is not available in the scripts path. Available tasks are: {self._available_tasks}")
        elif downstream_task_types is None:
            self.downstream_task_types = self._available_tasks
            self.downstream_task_type_path = [self.scripts_path / item for item in self.downstream_task_types]
        elif isinstance(downstream_task_types, str):
            self.downstream_task_types = [downstream_task_types]
            self.downstream_task_type_path = [self.scripts_path / downstream_task_types]
        elif isinstance(downstream_task_types, List):
            self.downstream_task_types = downstream_task_types.copy()
            self.downstream_task_type_path = [self.scripts_path / item for item in self.downstream_task_types]
        else:
            raise ValueError("Invalid downstream task types")

    @property
    def project_data_path(self):
        """The default data path is the project root/data. Override this method if you want to change the default data path."""
        return self.project_root / "data"

    @property
    def processed_data_path(self):
        """The default processed data path is the project root/data_processed. Override this method if you want to change the default processed data path."""
        return self.project_root / "data_processed"

    @property
    def scripts_path(self):
        """The default scripts path is the project root/scripts. Override this method if you want to change the default scripts path."""
        return self.dataset_path / "scripts"

    @property
    def annotations_path(self):
        """The default annotations path is the project root/annotations. Override this method if you want to change the default annotations path."""
        return self.dataset_path / "annotations"

    @property
    def files_path(self):
        """The default files path is the project root/files. Override this method if you want to change the default files path."""
        return self.dataset_path / "files"
