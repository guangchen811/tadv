from cadv_exploration.utils import load_dotenv

load_dotenv()

from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.utils import get_project_root

from scripts.python.utils import setup_logger, load_train_and_test_spark_data


def run_deequ_dv(data_name, processed_data_idx):
    logger = setup_logger("./deequ.log")
    dq_manager = DeequDataQualityManager()

    processed_data_path = get_project_root() / "data_processed" / f"{data_name}" / f"{processed_data_idx}"

    result_path = processed_data_path / "constraints" / "deequ_constraints.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    spark_train_data, spark_train, spark_validation_data, spark_validation = load_train_and_test_spark_data(
        data_name=data_name, processed_data_idx=processed_data_idx, dq_manager=dq_manager
    )

    constraints = dq_manager.get_constraints_for_spark_df(spark_train, spark_train_data, spark_validation,
                                                          spark_validation_data)
    constraints.save_to_yaml(result_path)

    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_validation.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()
    spark_validation.stop()


if __name__ == "__main__":
    run_deequ_dv(data_name="healthcare_dataset", processed_data_idx=0)
