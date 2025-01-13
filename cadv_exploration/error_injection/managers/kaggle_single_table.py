from pathlib import Path
from typing import Union

from cadv_exploration.error_injection.abstract_error_injection_manager import AbstractErrorInjectionManager
from cadv_exploration.loader import FileLoader
from cadv_exploration.loader.dataset.kaggle_dataset_loader import KaggleDatasetLoader


class KaggleSingleTableErrorInjectionManager(AbstractErrorInjectionManager):
    def __init__(self,
                 raw_file_path: Path,
                 target_table_name: str,
                 target_column_name: str,
                 processed_data_dir: Path,
                 submission_default_value: Union[str, float]
                 ):
        self.post_corruption_test_data = None
        self.raw_file_path = raw_file_path
        self.target_table_name = target_table_name
        self.target_column_name = target_column_name
        self.processed_data_dir = processed_data_dir
        self.submission_default_value = submission_default_value

        self.processed_data_path = self._create_processed_data_path()
        full_data = self.load_data()
        (
            self.train_data,
            self.validation_data,
            self.test_data,
            self.ground_truth,
            self.sample_submission
        ) = self._split_dataset_for_kaggle(
            full_data)

    def load_data(self):
        return FileLoader.load_csv(self.raw_file_path / f"{self.target_table_name}.csv")

    def error_injection(self, corrupts):
        post_corruption_test_data = self.test_data.copy(deep=True)
        for corrupt in corrupts:
            post_corruption_test_data = corrupt.transform(post_corruption_test_data)
        self.post_corruption_test_data = post_corruption_test_data

    def save_data(self):
        KaggleDatasetLoader().save_data(
            self.processed_data_path,
            self.train_data,
            self.validation_data,
            self.test_data,
            self.post_corruption_test_data,
            self.ground_truth,
            self.sample_submission
        )

    def _create_processed_data_path(self):
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_idx = len(list(self.processed_data_dir.iterdir()))
        processed_data_path = self.processed_data_dir / f"{processed_data_idx}"
        processed_data_path.mkdir(parents=True, exist_ok=True)
        return processed_data_path

    def _split_dataset_for_kaggle(self, full_data):
        full_data.drop(columns=["id"], inplace=True)
        full_data_shuffled = full_data.sample(frac=1, random_state=1).reset_index(drop=True)
        train_data_size = int(len(full_data_shuffled) * 0.6)
        validation_data_size = int(len(full_data_shuffled) * 0.1)
        train_data = full_data_shuffled[:train_data_size].reset_index(drop=True)
        validation_data = full_data_shuffled[train_data_size:train_data_size + validation_data_size].reset_index(
            drop=True)
        test_data = full_data_shuffled[train_data_size + validation_data_size:].reset_index(drop=True)

        train_data.reset_index(drop=False, inplace=True)
        train_data.rename(columns={"index": "id"}, inplace=True)
        validation_data.reset_index(drop=False, inplace=True)
        validation_data.rename(columns={"index": "id"}, inplace=True)
        test_data.reset_index(drop=False, inplace=True)
        test_data.rename(columns={"index": "id"}, inplace=True)
        validation_data['id'] = validation_data['id'] + len(train_data)
        test_data['id'] = test_data['id'] + len(train_data) + len(validation_data)

        ground_truth = test_data[["id", self.target_column_name]].copy()
        sample_submission = test_data[["id"]].copy()
        sample_submission[self.target_column_name] = self.submission_default_value
        test_data.drop(columns=[self.target_column_name], inplace=True)

        return train_data, validation_data, test_data, ground_truth, sample_submission
