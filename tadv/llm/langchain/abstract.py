from abc import ABC, abstractmethod


class AbstractLangChainTADV(ABC):
    @abstractmethod
    def invoke(self, input_variables: dict, max_retries: int):
        raise NotImplementedError


class SequentialLangChainTADV(AbstractLangChainTADV):
    """
    Abstract subclass for LangChain pipelines that execute tasks in a fixed, linear sequence of stages.
    Designed for workflows like data validation that require multiple, ordered processing steps.
    """

    @abstractmethod
    def single_invoke(self, input_variables: dict, num_stages: int):
        """
        Execute a single stage of the pipeline. Designed for retrying multiple times.
        """
        raise NotImplementedError

    @abstractmethod
    def show_prompts(self, input_variables: dict, num_stages: int):
        """
        Return the prompts used in the pipeline.
        """
        raise NotImplementedError

    @abstractmethod
    def invoke(self, input_variables: dict, num_stages: int = 3, max_retries: int = 3):
        """
        Orchestrates a linear multi-stage pipeline with retry support.

        Parameters:
            input_variables (dict): The input data required by the pipeline.
            num_stages (int): Number of sequential stages to execute.
            max_retries (int): Maximum number of retry attempts on failure.
        """
        raise NotImplementedError
