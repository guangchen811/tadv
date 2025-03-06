from abc import ABC, abstractmethod


class AbstractLangChainCADV(ABC):
    @abstractmethod
    def single_invoke(self, input_variables: dict, num_stages: int):
        raise NotImplementedError

    @abstractmethod
    def invoke(self, input_variables: dict, num_stages: int, max_retries: int):
        raise NotImplementedError
