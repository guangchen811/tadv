from tadv.data_models.expectation_config import ExpectationConfig


def test_from_yaml_file(gx_expectation_path):
    for config_file in gx_expectation_path.glob("*.yaml"):
        print(config_file)
        expectation_config = ExpectationConfig.from_yaml_file(config_file)
