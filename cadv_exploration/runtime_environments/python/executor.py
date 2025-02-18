import platform
import subprocess
import venv
from pathlib import Path

from cadv_exploration.runtime_environments.basis import ExecutorBase
from cadv_exploration.utils import get_current_folder


class PythonExecutor(ExecutorBase):
    env_path = get_current_folder() / "env"
    requirements_path = get_current_folder() / "requirements.txt"

    def __init__(self):
        super().__init__()
        self.env_path.mkdir(exist_ok=True)
        self.python_executable = self._get_python_executable()

    def run(self, project_name: str, input_path: Path, script_path: Path, output_path: Path, timeout: int = 120):
        print(f"Running Python script {script_path} with input {input_path} and output {output_path}")
        command = [str(self.python_executable), str(script_path), "--input", str(input_path), "--output",
                   str(output_path)]
        try:
            result = subprocess.run(command, check=True, timeout=timeout, capture_output=True, text=True)
            print(f"Python script {script_path} finished successfully\noutput files: {list(output_path.iterdir())}")
            return result.stdout  # Return standard output
        except subprocess.CalledProcessError as e:
            print(f"Error running script {script_path}: {e.stderr}")
            return f"Error: {e.stderr}"  # Return the error message
        except subprocess.TimeoutExpired:
            print(f"Script {script_path} timed out after {timeout} seconds.")
            return f"Error: Script {script_path} timed out after {timeout} seconds."

    def run_script(self, project_name: str, input_path: Path, script_context: str, output_path: Path,
                   timeout: int = 120):
        print(f"Running Python script with context {script_context} with input {input_path} and output {output_path}")
        script_path = get_current_folder() / "script.py"
        script_path.write_text(script_context)
        command = [str(self.python_executable), str(script_path), "--input", str(input_path), "--output",
                   str(output_path)]
        try:
            result = subprocess.run(command, check=True, timeout=timeout, capture_output=True, text=True)
            print(f"Python script with context finished successfully\noutput files: {list(output_path.iterdir())}")
            return result.stdout  # Return standard output
        except subprocess.CalledProcessError as e:
            print(f"Error running script with context: {e.stderr}")
            return f"Error: {e.stderr}"

    def _get_python_executable(self):
        self._create_or_update_environment()
        if platform.system() == "Windows":
            python_executable = self.env_path / "Scripts" / "python.exe"
        else:
            python_executable = self.env_path / "bin" / "python"
        return python_executable

    def _create_or_update_environment(self):
        pyvenv_cfg = self.env_path / "pyvenv.cfg"
        if not pyvenv_cfg.exists():
            self._create_environment()
        if not self._check_env_against_requirements():
            self._update_environment()

    def _create_environment(self):
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(self.env_path)
        pip_path = self._get_pip_path()
        print(f"Installing requirements from {self.requirements_path} into {self.env_path}")
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        subprocess.check_call([str(pip_path), "install", "-r", str(self.requirements_path)])

    def _update_environment(self):
        pip_path = self._get_pip_path()
        print(f"Updating requirements from {self.requirements_path} into {self.env_path}")
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        subprocess.check_call([str(pip_path), "install", "-r", str(self.requirements_path)])

    def _check_env_against_requirements(self):
        reqs = self.requirements_path.read_text().split("\n")
        reqs = [req.split("==") for req in reqs if req]
        reqs = [tuple(req) if len(req) == 2 else req[0] for req in reqs]
        if not reqs:
            return True
        pip_path = self._get_pip_path()
        installed = subprocess.check_output([str(pip_path), "freeze"]).decode("utf-8").split("\n")
        installed = [req.split("==") for req in installed]
        installed_with_version = [tuple(i) if len(i) == 2 else i[0] for i in installed]
        installed_wo_version = [i[0] for i in installed]
        return all((req in installed_with_version or req in installed_wo_version) for req in reqs)

    def _get_pip_path(self):
        if platform.system() == "Windows":
            pip_path = self.env_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.env_path / "bin" / "pip"
        return pip_path
