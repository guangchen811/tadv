from cadv_exploration.utils import load_dotenv

load_dotenv()

from cadv_exploration.dq_manager import DeequDataQualityManager
from cadv_exploration.loader import FileLoader
from utils import get_project_root


def main():
    dq_manager = DeequDataQualityManager()
    project_root = get_project_root()
    processed_data_path = project_root / "data_processed" / "playground-series-s4e10" / "0"
    train_data = FileLoader.load_csv(processed_data_path / "files_with_clean_test_data" / "train.csv")
    spark_train_data, spark_train = dq_manager.spark_df_from_pandas_df(train_data)
    check_strings = [
        ".hasMin('person_age', lambda x: x > 18)",
        ".hasMax('person_age', lambda x: x < 120)",
        ".isComplete('loan_status')",
        ".hasCompleteness('loan_status', lambda x: x == 1.0)",
        ".isUnique('id')",
        ".hasUniqueValueRatio(['id'], lambda x: x > 0.8)",
        ".hasEntropy('loan_status', lambda x: x > 0.4)",
        ".hasMutualInformation('loan_grade', 'loan_amnt', lambda x: x < 0.1)",
        ".hasApproxQuantile('person_income', 0.5, lambda x: x > 0.8)",
        ".hasMinLength('loan_intent', lambda x: x > 1)",
        ".hasMaxLength('loan_intent', lambda x: x < 20)",
        ".hasStandardDeviation('person_income', lambda x: x > 0.8)",
        ".hasApproxCountDistinct('loan_intent', lambda x: x > 0.8)",
        ".hasCorrelation('person_income', 'loan_amnt', lambda x: x > 0.3)",
        ".satisfies('person_income > 0 WHERE loan_amnt > 0', lambda x: x > 0.8)",
        ".hasPattern('person_home_ownership', 'RENT|OWN|MORTGAGE|OTHER', lambda x: x > 0.8)",
        ".isContainedIn('loan_grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])",
        ".containsURL('loan_intent', lambda x: x == 0)",
        ".isPositive('person_income')",
        ".isGreaterThan('person_income', 'loan_amnt', lambda x: x > 0.8)",
    ]
    check_results = dq_manager.apply_checks_from_strings(spark_train, spark_train_data,
                                                            check_strings)
    for check_result in check_results:
        print(check_result)
    spark_train.sparkContext._gateway.shutdown_callback_server()
    spark_train.stop()


if __name__ == "__main__":
    main()
