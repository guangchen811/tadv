from tadv.utils import load_dotenv

load_dotenv()

from tadv.dq_manager import DeequDataQualityManager
from tadv.utils import get_project_root

from scripts.python.utils import setup_logger, load_train_and_test_spark_data


def run_deequ_dv(dataset_name, downstream_task, processed_data_label):
    logger = setup_logger(get_project_root() / "logs" / "deequ_dv.log")
    dq_manager = DeequDataQualityManager()

    processed_data_path = get_project_root() / "data_processed" / dataset_name / downstream_task / f"{processed_data_label}"

    result_path = processed_data_path / "constraints" / "deequ_constraints.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    spark_train_data, spark_train, spark_validation_data, spark_validation = load_train_and_test_spark_data(
        dataset_name=dataset_name, downstream_task=downstream_task, processed_data_label=processed_data_label,
        dq_manager=dq_manager
    )

    constraints = dq_manager.get_constraints_for_spark_df(spark_train, spark_train_data, spark_validation,
                                                          spark_validation_data)
    constraints.save_to_yaml(result_path)

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    dataset_name_options = ["playground-series-s4e10", "healthcare_dataset"]
    downstream_task_options = ["ml_inference_classification", "ml_inference_regression", "sql_query",
                               "webpage_generation"]

    run_deequ_dv(dataset_name=dataset_name_options[0],
                 downstream_task=downstream_task_options[0],
                 processed_data_label="0"
                 )
