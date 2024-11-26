from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root
from error_injection.corrupts import *


# This case can be treated as context information for ml inference. (the test data needs to be validated)
def error_injection():
    target_column = "loan_status"
    sample_default_value = 0.5

    project_root = get_project_root()
    local_data_path = project_root / "data" / "playground-series-s4e10"
    file_path = local_data_path / "files"
    processed_data_dir = project_root / "data_processed" / "playground-series-s4e10"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_idx = len(list(processed_data_dir.iterdir()))
    processed_data_path = processed_data_dir / f"{processed_data_idx}"
    processed_data_path.mkdir(parents=True, exist_ok=True)

    full_train_data = FileLoader.load_csv(file_path / "train.csv")
    full_train_data.drop(columns=["id"], inplace=True)

    train_data, validation_data, test_data = split_dataset(full_train_data)

    train_data.reset_index(drop=False, inplace=True)
    train_data.rename(columns={"index": "id"}, inplace=True)
    validation_data.reset_index(drop=False, inplace=True)
    validation_data.rename(columns={"index": "id"}, inplace=True)
    test_data.reset_index(drop=False, inplace=True)
    test_data.rename(columns={"index": "id"}, inplace=True)
    validation_data['id'] = validation_data['id'] + len(train_data)
    test_data['id'] = test_data['id'] + len(train_data) + len(validation_data)

    ground_truth = test_data[["id", target_column]].copy()
    sample_submission = test_data[["id"]].copy()
    sample_submission[target_column] = sample_default_value
    test_data.drop(columns=[target_column], inplace=True)

    # Inject errors on the test data
    error_injectors = build_error_injectors()

    post_corruption_test_data = test_data.copy(deep=True)
    for error_injector in error_injectors:
        post_corruption_test_data = error_injector.transform(post_corruption_test_data)

    files_with_clean_test_data_path = processed_data_path / "files_with_clean_test_data"
    files_with_corrupted_test_data_path = processed_data_path / "files_with_corrupted_test_data"
    files_with_clean_test_data_path.mkdir(parents=True, exist_ok=True)
    files_with_corrupted_test_data_path.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(files_with_clean_test_data_path / "train.csv", index=False)
    validation_data.to_csv(files_with_clean_test_data_path / "validation.csv", index=False)
    test_data.to_csv(files_with_clean_test_data_path / "test.csv", index=False)
    ground_truth.to_csv(files_with_clean_test_data_path / "ground_truth.csv", index=False)
    sample_submission.to_csv(files_with_clean_test_data_path / "sample_submission.csv", index=False)

    train_data.to_csv(files_with_corrupted_test_data_path / "train.csv", index=False)
    validation_data.to_csv(files_with_corrupted_test_data_path / "validation.csv", index=False)
    post_corruption_test_data.to_csv(files_with_corrupted_test_data_path / "test.csv", index=False)
    ground_truth.to_csv(files_with_corrupted_test_data_path / "ground_truth.csv", index=False)
    sample_submission.to_csv(files_with_corrupted_test_data_path / "sample_submission.csv", index=False)


def build_error_injectors():
    error_injectors = []
    error_injectors.append(Scaling(columns=['loan_amnt'], severity=0.2))
    error_injectors.append(MissingCategoricalValueCorruption(columns=['person_home_ownership'], severity=0.1,
                                                             corrupt_strategy="to_majority"))
    error_injectors.append(MissingCategoricalValueCorruption(columns=['cb_person_default_on_file'], severity=0.1,
                                                             corrupt_strategy="to_random"))
    error_injectors.append(GaussianNoise(columns=['person_income'], severity=0.2))
    error_injectors.append(GaussianNoise(columns=['person_emp_length'], severity=0.2))
    error_injectors.append(ColumnInserting(columns=['loan_intent'], severity=0.1, corrupt_strategy="add_prefix"))
    error_injectors.append(
        ColumnInserting(columns=['person_home_ownership', 'person_age'], severity=0.1, corrupt_strategy="concatenate"))
    error_injectors.append(MaskValues(columns=['loan_grade'], severity=0.1))
    return error_injectors


def split_dataset(full_data):
    full_data_shuffled = full_data.sample(frac=1, random_state=1).reset_index(drop=True)
    train_data_size = int(len(full_data_shuffled) * 0.6)
    validation_data_size = int(len(full_data_shuffled) * 0.1)
    train_data = full_data_shuffled[:train_data_size].reset_index(drop=True)
    validation_data = full_data_shuffled[train_data_size:train_data_size + validation_data_size].reset_index(
        drop=True)
    test_data = full_data_shuffled[train_data_size + validation_data_size:].reset_index(drop=True)

    return train_data, validation_data, test_data


if __name__ == "__main__":
    error_injection()
