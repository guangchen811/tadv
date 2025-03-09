import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.python.relevant_column_detection.metrics import RelevantColumnDetectionMetric
from scripts.python.relevant_column_detection.run_pipeline import task_group_mapping
from scripts.python.utils import load_previous_and_new_spark_data
from tadv.dq_manager import DeequDataQualityManager
from tadv.utils import get_project_root, get_current_folder, get_task_instance


def result_calculation(dataset_name, model_name, processed_data_label):
    original_data_path = get_project_root() / "data" / f"{dataset_name}"
    dq_manager = DeequDataQualityManager()

    metric_evaluator = RelevantColumnDetectionMetric(average='macro')
    result_each_type = {}

    spark_previous_data, spark_previous, _, _ = load_previous_and_new_spark_data(
        dataset_name=dataset_name,
        downstream_task="sql_query",
        processed_data_label=processed_data_label,
        dq_manager=dq_manager
    )

    column_list = sorted(spark_previous_data.columns, key=lambda x: x.lower())

    result_path = get_current_folder() / "relevant_columns" / f"{dataset_name}" / f"{model_name}"
    result_path.mkdir(parents=True, exist_ok=True)
    for task_type in ['bi', 'dev', 'feature_engineering', 'classification', 'regression', 'info']:
        scripts_path_dir = original_data_path / "scripts" / task_group_mapping[task_type]
        print(task_type, end=' ')
        all_ground_truth_vectors = []
        all_relevant_columns_vectors = []
        for script_path in sorted(scripts_path_dir.iterdir(), key=lambda x: x.name):
            if task_type not in script_path.name and task_type != 'info':
                continue
            task_instance = get_task_instance(script_path)

            load_dir = result_path / f"relevant_columns__{script_path.stem}.txt"
            with open(load_dir, 'r') as f:
                relevant_columns_list = f.read().splitlines()

            ground_truth = sorted(task_instance.annotations['required_columns'], key=lambda x: x.lower())
            ground_truth_vector, relevant_columns_vector = metric_evaluator.binary_vectorize(column_list,
                                                                                             ground_truth,
                                                                                             relevant_columns_list)
            all_ground_truth_vectors.append(ground_truth_vector)
            all_relevant_columns_vectors.append(relevant_columns_vector)
        result_each_type[task_type] = [all_ground_truth_vectors, all_relevant_columns_vectors]
    return result_each_type


if __name__ == '__main__':
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    model_names = ["string-matching", "gpt-3.5-turbo", "gpt-4o", "gpt-4.5-preview"]
    processed_data_label = '0'

    task_official_name_mapping = {
        "bi": "Biz Intelligence",
        "dev": "Development",
        "feature_engineering": "Feature Eng.",
        "classification": "Classif.",
        "regression": "Regression",
        "info": "Web Gen."
    }

    # Set up the result path
    result_path = get_current_folder() / "figs"
    result_path.mkdir(parents=True, exist_ok=True)

    # Set up professional style
    sns.set_theme(style="whitegrid", font_scale=1.5, rc={"axes.labelsize": 14, "axes.titlesize": 16})

    # Create figure with two subplots (vertical layout)
    fig, axs = plt.subplots(nrows=2, figsize=(12, 6), sharex=False)

    for i, dataset_name in enumerate(dataset_name_options):
        all_results = {}
        for model_name in model_names:
            results = result_calculation(dataset_name, model_name, processed_data_label)
            all_results[model_name] = results

        plot_data = []
        for task_type in list(all_results[model_names[0]].keys()):
            for model_name in model_names:
                f1_scores = RelevantColumnDetectionMetric().statistics_calculation(
                    all_results[model_name][task_type][0],
                    all_results[model_name][task_type][1]
                )
                for score in f1_scores['f1_list']:
                    plot_data.append({
                        "Task Type": task_official_name_mapping[task_type],
                        "Model": model_name,
                        "F1 Score": score
                    })

        df = pd.DataFrame(plot_data)

        ax = sns.boxplot(
            x="Task Type", y="F1 Score", hue="Model",
            data=df, width=0.6, linewidth=1.5, ax=axs[i]
        )

        # Set subplot aesthetics
        ax.set_ylabel("F1 Score", fontsize=16, labelpad=10)
        ax.set_title(f"F1 Score Comparison on {dataset_name}", fontsize=18, pad=15)

        if i == 0:
            ax.legend(title="Model", title_fontsize=14, fontsize=12, loc="lower left", frameon=True)
        else:
            ax.legend_.remove()

        # Explicitly set x-tick labels only on the bottom plot
        if i == len(dataset_name_options) - 1:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="center", fontsize=14)
            ax.set_xlabel("Task Type", fontsize=16, labelpad=10)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])  # Remove x labels from upper subplot

        # Grid styling
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Fix layout
    plt.tight_layout()

    # Save figure in high resolution
    plt.savefig(result_path / "f1_score_boxplot_combined.png", dpi=300, bbox_inches="tight")
