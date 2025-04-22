from abc import ABC, abstractmethod

from tadv.data_models import Constraints


class ConstraintSuggesting(ABC):
    @abstractmethod
    def inference_constraints_for_spark_df(self, spark, spark_df, spark_validation=None,
                                           spark_validation_df=None) -> Constraints:
        """
        Infer constraints for the Spark DataFrame based on the provided validation DataFrame.
        """
        raise NotImplementedError("Subclasses should implement this method.")
