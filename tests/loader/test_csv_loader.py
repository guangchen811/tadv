from cadv_exploration.loader import FileLoader


def test_load_csv(resources_path):
    """Test the load_csv function."""
    df = FileLoader.load_csv(resources_path / "example_dataset_1" / "files" / "example_table.csv")
    assert df.shape[0] == 5
    assert df.shape[1] == 4


def test_load_csvs(resources_path):
    """Test the load_csvs function."""
    dfs = FileLoader.load_csvs(resources_path / "example_dataset_1" / "files")
    assert len(dfs) == 1
    assert dfs[0].shape[0] == 5
    assert dfs[0].shape[1] == 4
