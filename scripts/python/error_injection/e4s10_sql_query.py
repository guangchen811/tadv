from cadv_exploration.error_injection.managers.sql_query import GeneralErrorInjectionManager
from cadv_exploration.utils import get_project_root


def error_injection():
    project_root = get_project_root()
    error_injection_manager = GeneralErrorInjectionManager(
        raw_file_path=project_root / "data" / "playground-series-s4e10" / "files",
        target_table_name="train",
        processed_data_dir=project_root / "data_processed" / "playground-series-s4e10_sql_query",
        sample_size=1.0
    )

    corrupts = build_corrupts()

    error_injection_manager.error_injection(corrupts)

    error_injection_manager.save_data()


def build_corrupts():
    corrupts = []
    return corrupts


if __name__ == "__main__":
    error_injection()
