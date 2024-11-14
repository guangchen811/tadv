from loader import load_csv
from utils import get_project_root
import oyaml as yaml
import matplotlib.pyplot as plt


def plot(plt=None):
    project_root = get_project_root()
    local_data_path = project_root / "data" / "playground-series-s4e10"
    ground_truth = load_csv(local_data_path / "files_with_clean_test_data" / "ground_truth.csv")
    scripts_output_dir = local_data_path / "output"
    clean_test_data = load_csv(local_data_path / "files_with_clean_test_data" / "test.csv")
    corrupted_test_data = load_csv(local_data_path / "files_with_corrupted_test_data" / "test.csv")
    for script_output_dir in scripts_output_dir.iterdir():
        print(f"evaluating script: {script_output_dir.name}")

        cadv_suggestion_file_path = script_output_dir / "cadv_constraints.yaml"
        with open(cadv_suggestion_file_path, "r") as file:
            cadv_suggestions = yaml.safe_load(file)
        column_name = {column_name: len(suggestions)
                       for column_name, suggestions in cadv_suggestions['constraints'].items()}

        # plot x: column_name, y: number of suggestions
        plt.plot(column_name.keys(), column_name.values())
        plt.show()


if __name__ == "__main__":
    plot()
