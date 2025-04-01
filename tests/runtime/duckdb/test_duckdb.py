from pathlib import Path

from tadv.runtime_environments import DuckDBExecutor
from tadv.utils import get_project_root


def test_run_expected_output(tmp_path):
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


def test_run_with_invalid_sql(tmp_path):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    input_path = local_project_path / "files"

    # Create an invalid SQL file
    invalid_sql = "SELECT * FRM non_existent_table;"
    invalid_sql_path = tmp_path / "invalid.sql"
    invalid_sql_path.write_text(invalid_sql)

    output_path = tmp_path / "output"
    output_path.mkdir()

    output = executor.run(local_project_path.name, input_path, invalid_sql_path, output_path)

    # Check that output is None due to error
    assert output is None

    # Check that error.txt was created
    error_file = output_path / "error.txt"
    assert error_file.exists()

    # Optionally, check that the error message contains expected text
    error_content = error_file.read_text()
    assert "syntax error" in error_content.lower() or "parsing" in error_content.lower()


def test_run_script_executes_valid_sql(tmp_path):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    input_path = local_project_path / "files"
    output_path = tmp_path / "output"
    output_path.mkdir()

    sql_query = "SELECT COUNT(*) AS column_count FROM some_table;"  # replace `some_table` with an actual CSV stem

    # Dynamically extract one table name for reference in SQL
    table_name = next(input_path.iterdir()).stem
    sql_query = f"SELECT COUNT(*) AS row_count FROM {table_name};"

    output = executor.run_script(local_project_path.name, input_path, sql_query, output_path)

    assert output is not None
    assert "row_count" in output.columns
    assert output.iloc[0, 0] > 0  # At least one row expected


def test_run_script_handles_invalid_sql(tmp_path):
    executor = DuckDBExecutor()
    project_root = get_project_root()
    local_project_path = Path(
        project_root / "tests" / "resources" / "example_dataset_1"
    )
    input_path = local_project_path / "files"
    output_path = tmp_path / "output"
    output_path.mkdir()

    invalid_sql = "SELEC * FORM fake_table"

    output = executor.run_script(local_project_path.name, input_path, invalid_sql, output_path)

    assert output is None

    error_file = output_path / "error.txt"
    assert error_file.exists()
    assert "syntax error" in error_file.read_text().lower() or "parse" in error_file.read_text().lower()
