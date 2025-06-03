from tadv.data_models.expectation_schema import ExpectationSchema
from tadv.utils import get_project_root


class GXFunctionManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GXFunctionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        self.info_path = get_project_root() / "tadv" / "ir_translator" / "gx_expectations" / "expectations"
        self.info = self._get_expectation_info()
        self.__class__._initialized = True

    def _get_expectation_info(self):
        return [ExpectationSchema.from_yaml_file(config_file) for config_file in self.info_path.glob("*.yaml")]

    def get_all_text_descriptions(self):
        text_descriptions_list = [expectation.to_text_description() for expectation in self.info]
        return "----------\n".join(text_descriptions_list)

    def get_all_signatures(self):
        signatures_list = [expectation.to_signature() for expectation in self.info]
        return "----------\n".join(signatures_list)

    def get_expectation(self, name: str) -> ExpectationSchema:
        """
        Returns the expectation schema for the given expectation type.
        """
        for schema in self.info:
            if schema.Name == name:
                return schema
        raise ValueError(f"Expectation type '{name}' not found in info.")


if __name__ == "__main__":
    gx_function_manager = GXFunctionManager()
    # print(gx_function_manager.get_all_text_descriptions())
    # print(gx_function_manager.get_all_signatures())
    print(gx_function_manager.get_expectation("ExpectColumnMaxToBeBetween"))
