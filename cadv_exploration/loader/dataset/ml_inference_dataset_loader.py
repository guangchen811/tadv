class MLInferenceDatasetLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @staticmethod
    def save_data(processed_data_path, train_data, validation_data, test_data, post_corruption_test_data,
                  ground_truth, sample_submission):
        def save_files(base_path, target_test_data):
            base_path.mkdir(parents=True, exist_ok=True)
            train_data.to_csv(base_path / "train.csv", index=False)
            validation_data.to_csv(base_path / "validation.csv", index=False)
            target_test_data.to_csv(base_path / "test.csv", index=False)
            ground_truth.to_csv(base_path / "ground_truth.csv", index=False)
            sample_submission.to_csv(base_path / "sample_submission.csv", index=False)

        files_with_clean_test_data_path = processed_data_path / "files_with_clean_test_data"
        files_with_corrupted_test_data_path = processed_data_path / "files_with_corrupted_test_data"

        save_files(files_with_clean_test_data_path, test_data)
        save_files(files_with_corrupted_test_data_path, post_corruption_test_data)

    @staticmethod
    def load_data(processed_data_path, file_dir):
        from loader import FileLoader
        return (
            FileLoader.load_csv(processed_data_path / file_dir / "train.csv"),
            FileLoader.load_csv(processed_data_path / file_dir / "validation.csv"),
            FileLoader.load_csv(processed_data_path / file_dir / "test.csv"),
            FileLoader.load_csv(processed_data_path / file_dir / "ground_truth.csv"),
            FileLoader.load_csv(processed_data_path / file_dir / "sample_submission.csv")
        )
