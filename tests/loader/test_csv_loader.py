import os

from cadv_exploration.loader import FileLoader
from cadv_exploration.utils import get_project_root


def test_load_csv():
    """Test the load_csv function."""
    project_root = get_project_root()
    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "files"
    file_path = os.listdir(dir_path)[0]
    df = FileLoader.load_csv(dir_path / file_path)
    assert df.shape[0] == 55500
    assert df.shape[1] == 15


def test_load_csvs():
    """Test the load_csvs function."""
    project_root = get_project_root()
    dir_path = project_root / "data" / "prasad22" / "healthcare-dataset" / "files"
    dfs = FileLoader.load_csvs(dir_path)
    assert len(dfs) == 1
    assert dfs[0].shape[0] == 55500
    assert dfs[0].shape[1] == 15
