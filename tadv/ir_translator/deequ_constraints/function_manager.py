import oyaml as yaml

from tadv.data_models.deequ_schema import DeequSchema
from tadv.utils import get_project_root


class DeequFunctionManager:
    def __init__(self):
        self.info_path = get_project_root() / "tadv" / "ir_translator" / "deequ_constraints" / "info.yaml"
        self.info = self.get_info()

    def get_info(self):
        """
        Reads the info.yaml file and returns its content.
        """
        with open(self.info_path, 'r') as file:
            info = yaml.safe_load(file)
        return [DeequSchema.from_dict({k: v}) for k, v in info.items()]


if __name__ == "__main__":
    deequ_function_manager = DeequFunctionManager()
    info = deequ_function_manager.get_info()
    print(info)
    # Example usage: print the info
    # print(deequ_function_manager.get_info())
