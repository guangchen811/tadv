from pathlib import Path

import duckdb

from cadv_exploration.runtime_environments.basis import ExecutorBase


class DuckDBExecutor(ExecutorBase):
    def __init__(self):
        super().__init__()

    def run(self, project_name: str, input_path: Path, script_path: Path, output_path: Path, timeout: int = 120):
        csv_files = input_path.iterdir()
        db = duckdb.connect(':memory:')
        for csv_file in csv_files:
            db.sql(f"CREATE TABLE {csv_file.stem} AS SELECT * FROM read_csv_auto('{csv_file}')")
        try:
            output = db.sql(script_path.read_text()).fetchdf()
            print(f"Output:\n {output}")
            output.to_csv(output_path / "output.csv", index=False)
        except Exception as e:
            print(f"Error: {e}, writing error to {output_path / 'error.txt'}")
            with open(output_path / "error.txt", "w") as f:
                f.write(str(e))
            output = None
        db.close()
        return output

    def run_script(self, project_name: str, input_path: Path, script_context: str, output_path: Path,
                   timeout: int = 120):
        csv_files = input_path.iterdir()
        db = duckdb.connect(':memory:')
        for csv_file in csv_files:
            db.sql(f"CREATE TABLE {csv_file.stem} AS SELECT * FROM read_csv_auto('{csv_file}')")
        try:
            output = db.sql(script_context).fetchdf()
            print(f"Output:\n {output}")
            output.to_csv(output_path / "output.csv", index=False)
        except Exception as e:
            print(f"Error: {e}, writing error to {output_path / 'error.txt'}")
            with open(output_path / "error.txt", "w") as f:
                f.write(str(e))
            output = None
        db.close()
        return output
