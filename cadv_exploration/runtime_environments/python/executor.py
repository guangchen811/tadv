import platform
import subprocess
import venv
from pathlib import Path

from cadv_exploration.runtime_environments.basis import ExecutorBase
from utils import get_current_folder


class PythonExecutor(ExecutorBase):
    env_path = get_current_folder() / "env"
    requirements_path = get_current_folder() / "requirements.txt"
    python_version = "3.12"

    def __init__(self):
        super().__init__()
        self.env_path.mkdir(exist_ok=True)
        self.python_executable = self._get_python_executable()

    def run(self, project_name: str, input_path: Path, script_path: Path, output_path: Path, timeout: int = 120):
        print(f"Running Python script {script_path} with input {input_path} and output {output_path}")
        # command = [str(self.python_executable), str(script_path), str(input_path), str(output_path)]
        command = [str(self.python_executable), "-c", f"print('Hello, World!')"]
        subprocess.run(command, check=True, timeout=timeout)

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
        subprocess.check_call([str(pip_path), "install", "-r", str(self.requirements_path)])

    def _update_environment(self):
        pip_path = self._get_pip_path()
        print(f"Updating requirements from {self.requirements_path} into {self.env_path}")
        subprocess.check_call([str(pip_path), "install", "-r", str(self.requirements_path)])

    def _check_env_against_requirements(self):
        reqs = self.requirements_path.read_text().split("\n")
        if not reqs:
            return True
        pip_path = self._get_pip_path()
        installed = subprocess.check_output([str(pip_path), "freeze"]).decode("utf-8").split("\n")
        installed = [req.split("==")[0] for req in installed]
        return all(req in installed for req in reqs)

    def _get_pip_path(self):
        if platform.system() == "Windows":
            pip_path = self.env_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.env_path / "bin" / "pip"
        return pip_path
