from cadv_exploration.error_injection.managers.ml_inference import MLInferenceErrorInjectionManager

from cadv_exploration.utils import get_project_root
from error_injection.corrupts import *


# This case can be treated as context information for ml inference. (the test data needs to be validated)
def error_injection():
    project_root = get_project_root()
    error_injection_manager = MLInferenceErrorInjectionManager(
        raw_file_path=project_root / "data" / "playground-series-s4e10" / "files",
        target_table_name="train",
        target_column_name="loan_status",
        processed_data_dir=project_root / "data_processed" / "playground-series-s4e10_ml_inference",
        submission_default_value=0.5,
    )

    # Inject errors on the test data
    corrupts = build_corrupts()

    error_injection_manager.error_injection(corrupts)

    # Save the corrupted test data
    error_injection_manager.save_data()
    

def build_corrupts():
    corrupts = []
    corrupts.append(Scaling(columns=['loan_amnt'], severity=0.2))
    corrupts.append(MissingCategoricalValueCorruption(columns=['person_home_ownership'], severity=0.1,
                                                      corrupt_strategy="to_majority"))
    corrupts.append(MissingCategoricalValueCorruption(columns=['cb_person_default_on_file'], severity=0.1,
                                                      corrupt_strategy="to_random"))
    corrupts.append(GaussianNoise(columns=['person_income'], severity=0.2))
    corrupts.append(GaussianNoise(columns=['person_emp_length'], severity=0.2))
    corrupts.append(ColumnInserting(columns=['loan_intent'], severity=0.1, corrupt_strategy="add_prefix"))
    corrupts.append(
        ColumnInserting(columns=['person_home_ownership', 'person_age'], severity=0.1, corrupt_strategy="concatenate"))
    corrupts.append(MaskValues(columns=['loan_grade'], severity=0.1))
    corrupts.append(ColumnDropping(columns=['person_income'], severity=0.1))
    return corrupts


if __name__ == "__main__":
    error_injection()
