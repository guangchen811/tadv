from pathlib import Path

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


def test_python_executor_handles_script_error(tmp_path):
    executor = PythonExecutor()

    # Setup dummy input path (can be empty, since script will fail before using it)
    input_path = tmp_path / "input"
    input_path.mkdir()

    # Setup output path
    output_path = tmp_path / "output"
    output_path.mkdir()

    # Create a failing script (invalid Python code)
    failing_script = tmp_path / "failing_script.py"
    failing_script.write_text(textwrap.dedent("""
        import sys
        raise ValueError("Intentional error for testing")
    """))

    result = executor.run("test_project", input_path, failing_script, output_path)

    # Assert that an error message is returned
    assert result.startswith("Error:")
    assert "Intentional error for testing" in result

    # Assert that error.txt exists and contains the error
    error_file = output_path / "error.txt"
    assert error_file.exists()
    error_content = error_file.read_text()
    assert "Intentional error for testing" in error_content

    # Ensure no output.csv is created
    assert not (output_path / "output.csv").exists()


import pandas as pd
import textwrap


def test_run_script_executes_python_code_successfully(tmp_path):
    from tadv.runtime_environments import PythonExecutor

    executor = PythonExecutor()

    # Prepare input data
    input_path = tmp_path / "input"
    input_path.mkdir()
    df = pd.DataFrame({
        "FirstName": ["Alice", "Bob"],
        "LastName": ["Smith", "Jones"]
    })
    df.to_csv(input_path / "people.csv", index=False)

    # Prepare output path
    output_path = tmp_path / "output"
    output_path.mkdir()

    # Python script as string (script_context)
    script_context = textwrap.dedent("""
        import pandas as pd
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str)
        parser.add_argument("--output", type=str)
        args = parser.parse_args()

        input_path = Path(args.input)
        df = pd.read_csv(input_path / "people.csv")
        df["FullName"] = df["FirstName"] + " " + df["LastName"]
        df.to_csv(Path(args.output) / "output.csv", index=False)
    """)

    result = executor.run_script("test_project", input_path, script_context, output_path)

    # Validate that output.csv was created and has the expected data
    output_csv = output_path / "output.csv"
    assert output_csv.exists()

    df_output = pd.read_csv(output_csv)
    assert "FullName" in df_output.columns
    assert df_output["FullName"].tolist() == ["Alice Smith", "Bob Jones"]
    assert "Error:" not in result
