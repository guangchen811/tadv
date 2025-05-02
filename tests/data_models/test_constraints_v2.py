from tadv.data_models import ConstraintsWithSources


def test_from_yaml(constraints_with_sources_instance, tmp_path):
    constraints_with_sources_instance.save_to_yaml(str(tmp_path / "constraints.yaml"))
    constraints = ConstraintsWithSources.from_yaml(str(tmp_path / "constraints.yaml"))

    assert constraints.to_dict() == constraints_with_sources_instance.to_dict()
