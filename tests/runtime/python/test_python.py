from pathlib import Path

import pandas as pd

from tadv.runtime_environments import PythonExecutor
from tadv.utils import get_project_root


def test_runnable(tmp_path):
    executor = PythonExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    input_path = Path(
        local_project_path / "files"
    )
    script_dir = Path(
        local_project_path
        / "script_web"
        / "example_web_script.py"
    )
    output_path = tmp_path / "output"
    output_path.mkdir()
    executor.run(local_project_path.name, input_path, script_dir, output_path)
    assert len(list(output_path.iterdir())) == 1
    assert (output_path / "output.csv").exists()
    output_df = pd.read_csv(output_path / "output.csv")
    assert output_df.shape == (5, 5)
    assert output_df["FullName"].tolist() == [
        "Rachel Booker",
        "Laura Grey",
        "Craig Johnson",
        "Mary Jenkins",
        "Jamie Smith",
    ]
