from pathlib import Path

from cadv_exploration.runtime_environments import PythonExecutor
from utils import get_project_root


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
    _ = executor.run(local_project_path.name, input_path, script_dir, output_path)
