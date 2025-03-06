from cadv_exploration.error_injection.corrupts import Scaling, MissingCategoricalValueCorruption, GaussianNoise, \
    ColumnInserting, \
    MaskValues, ColumnDropping
from cadv_exploration.error_injection.managers.sql_query import GeneralErrorInjectionManager
from cadv_exploration.utils import get_project_root


def error_injection():
    project_root = get_project_root()
    error_injection_manager = GeneralErrorInjectionManager(
        raw_file_path=project_root / "data" / "playground-series-s4e10" / "files",
        target_table_name="train",
        processed_data_dir=project_root / "data_processed" / "playground-series-s4e10_webpage_generation",
        sample_size=1.0
    )

    corrupts = build_corrupts()

    error_injection_manager.error_injection(corrupts)

    error_injection_manager.save_data()


def build_corrupts():
    corrupts = [Scaling(columns=['loan_amnt'], severity=0.2),
                MissingCategoricalValueCorruption(columns=['person_home_ownership'], severity=0.1,
                                                  corrupt_strategy="to_majority"),
                MissingCategoricalValueCorruption(columns=['cb_person_default_on_file'], severity=0.1,
                                                  corrupt_strategy="to_random"),
                GaussianNoise(columns=['person_income'], severity=0.2),
                GaussianNoise(columns=['person_emp_length'], severity=0.2),
                ColumnInserting(columns=['loan_intent'], severity=0.1, corrupt_strategy="add_prefix"),
                ColumnInserting(columns=['person_home_ownership', 'person_age'], severity=0.1,
                                corrupt_strategy="concatenate"), MaskValues(columns=['loan_grade'], severity=0.1),
                ColumnDropping(columns=['person_income'], severity=0.1)]
    return corrupts


if __name__ == "__main__":
    error_injection()
