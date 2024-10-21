import os
import subprocess
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class KaggleExecutor:
    def __init__(self):
        pass

    def run(self, local_data_path: Path, script_path: Path, output_path: Path, is_competition: bool):
        command_prefix = self._prepare_prefix(local_data_path, script_path, output_path, is_competition)
        script_file_name = os.path.basename(script_path)
        # script_file_path =
        if str(script_path).endswith(".py"):
            command = self._compsoe_py_command(command_prefix, script_file_name)
        elif str(script_path).endswith(".ipynb"):
            command = self._compsoe_ipynb_command(command_prefix, script_file_name)
        else:
            raise ValueError("Invalid script type")

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Execution successful!")
            print(result.stdout.decode())
        else:
            print("Execution failed:")
            print(result.stderr.decode())
        return result

    def _compsoe_py_command(self, command_prefix: list, script_file_name: str) -> list:
        # Run the Kaggle environment in Docker
        command = command_prefix + ["python", f"/kaggle/script/{script_file_name}"]
        return command

    def _compsoe_ipynb_command(self, command_prefix: list, script_file_name: str) -> list:
        # https://nbconvert.readthedocs.io/en/latest/execute_api.html# module-nbconvert.preprocessors
        command = command_prefix + ["jupyter", "nbconvert", "--to", "notebook", "--output",
                                    "/kaggle/output/output.ipynb", "--execute", f"/kaggle/script/{script_file_name}"]
        return command

    @staticmethod
    def _prepare_prefix(local_data_path: Path, script_path: Path, output_path: Path, is_competition):
        if is_competition:
            docker_data_path = f"/kaggle/input/{local_data_path.name}/"
        else:
            docker_data_path = f"/kaggle/input/{local_data_path.name}/"
        if str(script_path).endswith(".py"):
            script_type = "py"
        elif str(script_path).endswith(".ipynb"):
            script_type = "ipynb"
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{local_data_path / 'files'}:{docker_data_path}",
            "-v",
            f"{script_path.parent}:/kaggle/script/",
            "-v",
            f"{output_path}:/kaggle/output/",
            "kaggle-env/python:1.0.0",
        ]
        return command
