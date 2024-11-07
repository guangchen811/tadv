import numpy as np
import pandas as pd
from cadv_exploration.error_injection.corrupts import Scaling, MissingCategoricalValueCorruption
from cadv_exploration.loader import load_csv
from cadv_exploration.utils import get_project_root


def error_injection():
    project_root = get_project_root()
    file_path = (
            project_root
            / "data"
            / "prasad22"
            / "healthcare-dataset"
            / "files"
            / "healthcare_dataset.csv"
    )
    df = load_csv(file_path)
    original_df = df.sample(frac=0.7, random_state=1).reset_index(drop=True)
    original_train_df = original_df.sample(frac=0.9, random_state=1).reset_index(drop=True)
    original_validation_df = original_df.drop(original_train_df.index).reset_index(drop=True)
    pre_corruption_df = df.drop(original_df.index).reset_index(drop=True)

    # Inject errors
    scaler = Scaling(columns=['Age'], severity=0.2)
    missing_to_random = MissingCategoricalValueCorruption(columns=['Blood Type'], severity=0.1,
                                                          corrupt_strategy="to_random")
    missing_to_majority = MissingCategoricalValueCorruption(columns=['Medical Condition'], severity=0.1,
                                                            corrupt_strategy="to_majority")
    missing_to_remove = MissingCategoricalValueCorruption(columns=['Insurance Provider'], severity=0.1,
                                                          corrupt_strategy="remove")

    post_corruption_df = scaler.transform(pre_corruption_df)
    post_corruption_df = missing_to_random.transform(post_corruption_df)
    post_corruption_df = missing_to_majority.transform(post_corruption_df)
    post_corruption_df = missing_to_remove.transform(post_corruption_df)

    original_train_path = file_path.parent.parent / "broken_files" / "original_train"
    original_validation_path = file_path.parent.parent / "broken_files" / "original_validation"
    pre_corruption_path = file_path.parent.parent / "broken_files" / "pre_corruption"
    post_corruption_path = file_path.parent.parent / "broken_files" / "post_corruption"
    original_train_path.mkdir(parents=True, exist_ok=True)
    original_validation_path.mkdir(parents=True, exist_ok=True)
    pre_corruption_path.mkdir(parents=True, exist_ok=True)
    post_corruption_path.mkdir(parents=True, exist_ok=True)

    original_train_df.to_csv(original_train_path / "healthcare_dataset.csv", index=False)
    original_validation_df.to_csv(original_validation_path / "healthcare_dataset.csv", index=False)
    pre_corruption_df.to_csv(pre_corruption_path / "healthcare_dataset.csv", index=False)
    post_corruption_df.to_csv(post_corruption_path / "healthcare_dataset.csv", index=False)


if __name__ == "__main__":
    error_injection()
