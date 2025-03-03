from pathlib import Path

from project_manager.abstract import AbstractProjectManager


class ProjectManager(AbstractProjectManager):

    def __init__(self, project_root: Path = None, dataset_name: str = None, downstream_task_type: str = None):
        super().__init__(project_root=project_root, dataset_name=dataset_name)

        self._available_datasets = [item.name for item in self.project_data_root.iterdir()]

        self._available_tasks = []
        if not self.dataset_path.exists():
            raise ValueError(
                f"Dataset {dataset_name} is not available in the project data path. Available datasets are: {self._available_datasets}")
        if self.scripts_root.exists():
            self._available_tasks = [item.name for item in self.scripts_root.iterdir()]
        if downstream_task_type is not None and downstream_task_type not in self._available_tasks:
            raise ValueError(
                f"Downstream task type {downstream_task_type} is not available in the scripts path. Available tasks are: {self._available_tasks}")
        elif downstream_task_type is None:
            raise ValueError("Downstream task type cannot be None")
        elif isinstance(downstream_task_type, str):
            self.downstream_task_type = [downstream_task_type]
            self.downstream_task_type_paths = [self.scripts_root / downstream_task_type]
        else:
            raise ValueError("Invalid downstream task types, downstream_task_type should be a string")

    def get_result_path(self, subtask_name, version_name):
        return self.get_processed_data_path(subtask_name, version_name) / "output"

    def get_processed_data_path(self, subtask_name, version_name):
        downstream_processed_path_list = [item for sublist in self.downstream_processed_path_mapping.values() for item
                                          in (sublist if isinstance(sublist, list) else [sublist])]
        if subtask_name not in downstream_processed_path_list:
            raise ValueError(
                f"Subtask {subtask_name} is not available in the downstream task type {downstream_processed_path_list}")
        return self.processed_data_root / f"{subtask_name}_{subtask_name}" / f"{version_name}"

    def get_clean_new_data_path(self, subtask_name, version_name):
        return self.get_processed_data_path(subtask_name, version_name) / "files_with_clean_new_data"

    def get_corrupted_data_path(self, subtask_name, version_name):
        return self.get_processed_data_path(subtask_name, version_name) / "files_with_corrupted_data"



    def prepare_data_for_relevant_columns_detection(self, subtask_name):
        pass

    def prepare_data_for_constraints_inference(self, subtask_name):
        pass


if __name__ == '__main__':
    project_manager = ProjectManager(dataset_name="healthcare_dataset",
                                     downstream_task_type="web")
    project_manager.get_result_path(subtask_name="ml_inference_classification", version_name="base_version")
