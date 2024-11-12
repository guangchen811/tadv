from sklearn.metrics import roc_auc_score
import numpy as np
from cadv_exploration.loader import load_csv
from cadv_exploration.utils import get_project_root


def evaluate_playground_series_s4e10():
    project_root = get_project_root()
    local_data_path = project_root / "data" / "playground-series-s4e10"
    ground_truth = load_csv(local_data_path / "files" / "test.csv")
    scripts_output_dir = local_data_path / "output"
    for script_output_dir in scripts_output_dir.iterdir():
        prediction = load_csv(script_output_dir / "submission.csv")
        # Compare the results
        print(f"Comparing results for script: {script_output_dir.name}")
        print(f"AUC: {roc_auc_score(np.where(prediction['loan_status'].to_numpy() > 0.5, 1, 0), prediction['loan_status'].to_numpy())}")
        break


if __name__ == "__main__":
    evaluate_playground_series_s4e10()
