class SQLQueryDatasetLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @staticmethod
    def save_data(processed_data_path, previous_data, new_data, post_corruption_new_data):
        def save_files(base_path, target_new_data):
            base_path.mkdir(parents=True, exist_ok=True)
            previous_data.to_csv(base_path / "previous_data.csv", index=False)
            target_new_data.to_csv(base_path / "new_data.csv", index=False)

        files_with_clean_new_data_path = processed_data_path / "files_with_clean_new_data"
        files_with_corrupted_new_data_path = processed_data_path / "files_with_corrupted_new_data"

        save_files(files_with_clean_new_data_path, new_data)
        save_files(files_with_corrupted_new_data_path, post_corruption_new_data)

    @staticmethod
    def save_error_injection_config(processed_data_path, corrupts):
        import oyaml as yaml
        config_path = processed_data_path / "error_injection_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump([corrupt.to_dict() for corrupt in corrupts], f, default_flow_style=False)

    @staticmethod
    def load_data(processed_data_path, file_dir):
        from loader import FileLoader
        return (
            FileLoader.load_csv(processed_data_path / file_dir / "previous_data.csv"),
            FileLoader.load_csv(processed_data_path / file_dir / "new_data.csv"),
            FileLoader.load_csv(processed_data_path / file_dir / "post_corruption_new_data.csv")
        )
