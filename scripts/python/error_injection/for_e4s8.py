from error_injection import MissingCategoricalValueCorruption
from error_injection.corrupts import Scaling
from loader import load_csv
from utils import get_project_root


# This case can be treated as context information for ml inference. (the test data needs to be validated)
def error_injection():
    target_column = "class"
    sample_default_value = 'e'
    smapling_ratio = 0.2

    project_root = get_project_root()
    local_data_path = project_root / "data" / "playground-series-s4e8"
    file_path = local_data_path / "files"
    full_train_data = load_csv(file_path / "train.csv")
    sampled_data = full_train_data.sample(frac=smapling_ratio, random_state=1).reset_index(drop=True)
    sampled_data.drop(columns=["id"], inplace=True)

    train_data, validation_data, test_data = split_dataset(sampled_data)

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

    # # Inject errors on the test data
    # scaler = Scaling(columns=['person_age'], severity=0.2)
    # missing_to_majority = MissingCategoricalValueCorruption(columns=['person_home_ownership'], severity=0.1,
    #                                                         corrupt_strategy="to_majority")
    # missing_to_remove = MissingCategoricalValueCorruption(columns=['cb_person_default_on_file'], severity=0.1,
    #                                                       corrupt_strategy="to_random")

    post_corruption_test_data = test_data.copy()
    # post_corruption_test_data = scaler.transform(post_corruption_test_data)
    # post_corruption_test_data = missing_to_majority.transform(post_corruption_test_data)
    # post_corruption_test_data = missing_to_remove.transform(post_corruption_test_data)

    files_with_clean_test_data_path = local_data_path / "files_with_clean_test_data"
    files_with_corrupted_test_data_path = local_data_path / "files_with_corrupted_test_data"
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
