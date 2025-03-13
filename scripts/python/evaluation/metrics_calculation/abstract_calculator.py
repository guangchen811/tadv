from abc import abstractmethod


class AbstractMetricsCalculation():
    """
    Abstract class for metrics
    """

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, dataset_name, downstream_task, processed_data_label, script_output_dir):
        raise NotImplementedError
