from abc import abstractmethod, ABC, ABCMeta


class SingletonMeta(ABCMeta, type):
    """A metaclass for creating singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # If instance does not exist, create it and store in the dictionary
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class AbstractInspectorManager(ABC, metaclass=SingletonMeta):

    @abstractmethod
    def spark_df_to_column_desc(self, spark_df, spark):
        raise NotImplementedError
