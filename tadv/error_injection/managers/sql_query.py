from pathlib import Path

from tadv.error_injection.abstract_error_injection_manager import AbstractErrorInjectionManager
from tadv.loader import FileLoader
from tadv.loader.dataset.sql_query_dataset_loader import SQLQueryDatasetLoader


class GeneralErrorInjectionManager(AbstractErrorInjectionManager):
    def __init__(self,
                 raw_file_path: Path,
                 target_table_name: str,
                 processed_data_dir: Path,
                 sample_size: float
                 ):
        self.post_corruption_new_data = None
        self.raw_file_path = raw_file_path
        self.target_table_name = target_table_name
        self.processed_data_dir = processed_data_dir

        self.processed_data_path = self._create_processed_data_path(self.processed_data_dir)
        self.full_data = self.load_data()
        (
            self.previous_data,
            self.new_data
        ) = self._split_dataset(self.full_data, sample_size=sample_size)

    def load_data(self):
        return FileLoader.load_csv(self.raw_file_path / f"{self.target_table_name}.csv")

    def error_injection(self, corrupts):
        post_corruption_new_data = self.new_data.copy(deep=True)
        for corrupt in corrupts:
            post_corruption_new_data = corrupt.transform(post_corruption_new_data)
        self.post_corruption_new_data = post_corruption_new_data
        self.corrupts = corrupts

    def save_data(self):
        SQLQueryDatasetLoader().save_data(
            self.processed_data_path,
            self.previous_data,
            self.new_data,
            self.post_corruption_new_data
        )
        self.save_error_injection_config(self.corrupts)

    def save_error_injection_config(self, corrupts):
        SQLQueryDatasetLoader().save_error_injection_config(self.processed_data_path, corrupts)

    def _split_dataset(self, full_data, sample_size):
        full_data.drop(columns=["id"], inplace=True) if "id" in full_data.columns else None
        full_data_shuffled = full_data.sample(frac=sample_size, random_state=1).reset_index(drop=True)
        full_data_shuffled.reset_index(drop=False, inplace=True)
        full_data_shuffled.rename(columns={"index": "id"}, inplace=True)
        previous_data_size = int(len(full_data_shuffled) * 0.6)
        previous_data = full_data_shuffled[:previous_data_size].reset_index(drop=True)
        new_data = full_data_shuffled[previous_data_size:].reset_index(drop=True)
        return previous_data, new_data
