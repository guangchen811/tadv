from abc import ABC, abstractmethod
from pathlib import Path

from tadv.utils import get_project_root


class AbstractProjectManager(ABC):
    def __init__(self, project_root: Path = None, dataset_name: str = None):
        super().__init__()
        if project_root is None:
            self.project_root = get_project_root()
        else:
            self.project_root = project_root
        if dataset_name is None:
            raise ValueError('dataset_name cannot be None')
        else:
            self.dataset_path = self.project_data_root / dataset_name

    @property
    def project_data_root(self):
        """The default data path is the project root/data. Override this method if you want to change the default data path."""
        return self.project_root / "data"

    @property
    def processed_data_root(self):
        """The default processed data path is the project root/data_processed. Override this method if you want to change the default processed data path."""
        return self.project_root / "data_processed"

    @property
    def scripts_root(self):
        """The default scripts path is the project root/scripts. Override this method if you want to change the default scripts path."""
        return self.dataset_path / "scripts"

    @property
    def annotations_root(self):
        """The default annotations path is the project root/annotations. Override this method if you want to change the default annotations path."""
        return self.dataset_path / "annotations"

    @property
    def files_root(self):
        """The default files path is the project root/files. Override this method if you want to change the default files path."""
        return self.dataset_path / "files"

    @property
    def downstream_tasks_types(self):
        return ['ml_inference', 'sql_query', 'webpage_generation']

    def downstream_subtask_mapping(self, downstream_task_type):
        return {
            'ml_inference': ['classification', 'regression'],
            'sql_query': ['bi', 'dev', 'feature_engineering'],
            'webpage_generation': ['info']
        }[downstream_task_type]

    @property
    def downstream_processed_path_mapping(self):
        return {
            'ml_inference': ['ml_inference_classification', 'ml_inference_regression'],
            'sql_query': 'sql_query',
            'webpage_generation': 'webpage_generation'
        }

    @abstractmethod
    def get_processed_data_path(self, subtask_name, version_name):
        pass

    @abstractmethod
    def get_result_path(self, subtask_name, version_name):
        pass

    @abstractmethod
    def get_clean_new_data_path(self, subtask_name, version_name):
        pass

    @abstractmethod
    def get_corrupted_data_path(self, subtask_name, version_name):
        pass

    @abstractmethod
    def prepare_data_for_relevant_columns_detection(self, subtask_name):
        pass

    @abstractmethod
    def prepare_data_for_constraints_inference(self, subtask_name):
        pass
