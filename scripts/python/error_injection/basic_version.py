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
    original_part = df.sample(frac=0.7, random_state=1).reset_index(drop=True)
    before_broken_part = df.drop(original_part.index).reset_index(drop=True)

    # Inject errors
    scaler = Scaling(columns=['Age'], severity=0.2)
    missing_to_random = MissingCategoricalValueCorruption(columns=['Blood Type'], severity=0.1,
                                                          corrupt_strategy="to_random")
    missing_to_majority = MissingCategoricalValueCorruption(columns=['Medical Condition'], severity=0.1,
                                                            corrupt_strategy="to_majority")
    missing_to_remove = MissingCategoricalValueCorruption(columns=['Insurance Provider'], severity=0.1,
                                                          corrupt_strategy="remove")

    after_broken_part = scaler.transform(before_broken_part)
    after_broken_part = missing_to_random.transform(after_broken_part)
    after_broken_part = missing_to_majority.transform(after_broken_part)
    after_broken_part = missing_to_remove.transform(after_broken_part)

    original_path = file_path.parent.parent / "broken_files" / "original"
    before_broken_path = file_path.parent.parent / "broken_files" / "before_broken"
    after_broken_path = file_path.parent.parent / "broken_files" / "after_broken"
    original_path.mkdir(parents=True, exist_ok=True)
    before_broken_path.mkdir(parents=True, exist_ok=True)
    after_broken_path.mkdir(parents=True, exist_ok=True)

    original_part.to_csv(original_path / "healthcare_dataset.csv", index=False)
    before_broken_part.to_csv(before_broken_path / "healthcare_dataset.csv", index=False)
    after_broken_part.to_csv(after_broken_path / "healthcare_dataset.csv", index=False)


if __name__ == "__main__":
    error_injection()
