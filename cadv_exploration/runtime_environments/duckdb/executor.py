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
        output = db.sql(script_path.read_text()).fetchdf()
        return output
