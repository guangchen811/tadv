from tadv.dq_manager.abstract_data_quality_manager import AbstractDataQualityManager


class GreatExpectationQualityManager(AbstractDataQualityManager):
    def __init__(self):
        super().__init__()

    def apply_checks_from_strings(self, pandas_df, check_strings):
        pass
