import oyaml as yaml

from tadv.data_models.deequ_schema import DeequSchema
from tadv.utils import get_project_root


class DeequFunctionManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeequFunctionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        self.info_path = get_project_root() / "tadv" / "ir_translator" / "deequ_constraints" / "info.yaml"
        self.info = self.get_info()
        self.__class__._initialized = True

    def get_info(self):
        """
        Reads the info.yaml file and returns its content.
        """
        with open(self.info_path, 'r') as file:
            info = yaml.safe_load(file)
        return [DeequSchema.from_dict({k: v}) for k, v in info.items()]

    def get_constraint(self, name: str):
        """
        Returns the constraint schema for the given constraint type.
        """
        for schema in self.info:
            if schema.Name == name:
                return schema
        raise ValueError(f"Constraint type '{name}' not found in info.")


if __name__ == "__main__":
    deequ_function_manager = DeequFunctionManager()
    info = deequ_function_manager.get_info()
    print(info)
