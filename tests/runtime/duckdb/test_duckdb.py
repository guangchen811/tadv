from pathlib import Path

from tadv.runtime_environments import DuckDBExecutor
from tadv.utils import get_project_root


def test_runnable(tmp_path):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    input_path = Path(
        local_project_path / "files"
    )
    script_dir = Path(
        local_project_path
        / "script_sql"
        / "column_count.sql"
    )
    output_path = tmp_path / "output"
    output_path.mkdir()
    output = executor.run(local_project_path.name, input_path, script_dir, output_path)
    assert output is not None
    assert output.iloc[0, 0] == 5
