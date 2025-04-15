from tadv.data_models.expectation_config import ExpectationConfig


def test_from_yaml_file(gx_expectation_path):
    for config_file in gx_expectation_path.glob("*.yaml"):
        print(config_file)
        expectation_config = ExpectationConfig.from_yaml_file(config_file)


def test_to_text_description(gx_expectation_path):
    for config_file in gx_expectation_path.glob("*.yaml"):
        print(config_file)
        expectation_config = ExpectationConfig.from_yaml_file(config_file)
        description = expectation_config.to_text_description()
        print(description)


def test_to_signature(gx_expectation_path):
    for config_file in gx_expectation_path.glob("*.yaml"):
        print(config_file)
        expectation_config = ExpectationConfig.from_yaml_file(config_file)
        signature = expectation_config.to_signature()
        print(signature)
