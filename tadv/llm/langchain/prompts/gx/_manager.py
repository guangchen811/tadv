from tadv.data_models.expectation_config import ExpectationConfig
from tadv.utils import get_project_root


class GXConfigManager:
    def __init__(self):
        self.expectations_path = get_project_root() / "tadv" / "llm" / "langchain" / "prompts" / "gx" / "expectations"
        self.expectations = self._get_expectation_configs()

    def _get_expectation_configs(self):
        return [ExpectationConfig.from_yaml_file(config_file) for config_file in self.expectations_path.glob("*.yaml")]

    def get_all_text_descriptions(self):
        text_descriptions_list = [expectation.to_text_description() for expectation in self.expectations]
        return "----------\n".join(text_descriptions_list)

    def get_all_signatures(self):
        signatures_list = [expectation.to_signature() for expectation in self.expectations]
        return "----------\n".join(signatures_list)


if __name__ == "__main__":
    gx_config_manager = GXConfigManager()
    print(gx_config_manager.get_all_text_descriptions())
    print(gx_config_manager.get_all_signatures())
