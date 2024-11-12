import os
import signal
import subprocess
from pathlib import Path
import logging

import nbclient
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution time exceeded 2 minutes")


def extract_container_id(command):
    """Extract container ID from the command to reference it if needed."""
    try:
        return command[command.index("--name") + 1]  # Assuming `--name <container_id>` is in the command
    except (ValueError, IndexError):
        return None


class KaggleExecutor:
    def __init__(self):
        pass

    def run(self, local_project_path: Path, input_path: Path, script_path: Path, output_path: Path, timeout: int = 120):
        command_prefix = self._prepare_prefix(local_project_path, input_path, script_path, output_path)
        script_file_name = os.path.basename(script_path)
        # script_file_path =
        if str(script_path).endswith(".py"):
            command = self._compsoe_py_command(command_prefix, script_file_name)
        elif str(script_path).endswith(".ipynb"):
            command = self._compsoe_ipynb_command(command_prefix, script_file_name)
        else:
            raise ValueError("Invalid script type")
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(timeout)
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            signal.alarm(0)
        except TimeoutException as e:
            container_id = extract_container_id(command)
            if container_id:
                subprocess.run(["docker", "rm", "-f", container_id])
            print(f"Script {script_path} took too long to execute and was terminated.")
            return None

        if result.returncode == 0:
            print("Execution successful!")
        else:
            log_file = os.path.join(output_path, "error.log")
            logging.basicConfig(filename=log_file, level=logging.ERROR,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(f"{result.stderr.decode('utf-8')}")
            print(f"Execution failed! Writing the error to {log_file}")
        return result

    @staticmethod
    def _compsoe_py_command(command_prefix: list, script_file_name: str) -> list:
        # Run the Kaggle environment in Docker
        command = command_prefix + ["python", f"/kaggle/script/{script_file_name}"]
        return command

    @staticmethod
    def _compsoe_ipynb_command(command_prefix: list, script_file_name: str) -> list:
        # https://nbconvert.readthedocs.io/en/latest/execute_api.html# module-nbconvert.preprocessors
        command = command_prefix + ["jupyter", "nbconvert", "--to", "notebook", "--output",
                                    f"/kaggle/output/{script_file_name}", "--execute",
                                    f"/kaggle/script/{script_file_name}", "--allow-errors"]
        return command

    @staticmethod
    def _prepare_prefix(local_project_path: Path, input_path: Path, script_path: Path, output_path: Path):
        docker_data_path = f"/kaggle/input/{local_project_path.name}/"
        script_name = script_path.name.split(".")[0]
        if str(script_path).endswith(".py"):
            script_type = "py"
        elif str(script_path).endswith(".ipynb"):
            script_type = "ipynb"
        command = [
            "docker",
            "run",
            "--name",
            f"kaggle-{script_name}-executor",
            "--rm",
            "-v",
            f"{input_path}:{docker_data_path}",
            "-v",
            f"{script_path.parent}:/kaggle/script/",
            "-v",
            f"{output_path}:/kaggle/output/",
            "kaggle-env/python:1.0.0",
        ]
        return command
