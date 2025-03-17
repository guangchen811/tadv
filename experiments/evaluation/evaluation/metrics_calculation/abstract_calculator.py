from abc import abstractmethod


class AbstractMetricsCalculation():
    """
    Abstract class for metrics
    """

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, downstream_task, script_output_dir):
        raise NotImplementedError
