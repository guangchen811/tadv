import platform
from pathlib import Path
from unittest.mock import MagicMock

from tadv.runtime_environments import PythonExecutor


def test_get_python_executable_unix(monkeypatch):
    executor = PythonExecutor()

    # Mock platform.system to return "Linux"
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    # Mock _create_or_update_environment to avoid side effects
    executor._create_or_update_environment = MagicMock()

    # Set a fake env path
    executor.env_path = Path("/fake/env")

    expected = Path("/fake/env/bin/python")
    assert executor._get_python_executable() == expected


def test_get_python_executable_windows(monkeypatch):
    executor = PythonExecutor()

    # Mock platform.system to return "Windows"
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    # Mock _create_or_update_environment
    executor._create_or_update_environment = MagicMock()

    executor.env_path = Path("C:/fake/env")

    expected = Path("C:/fake/env/Scripts/python.exe")
    assert executor._get_python_executable() == expected


def test_get_pip_path_unix(monkeypatch):
    executor = PythonExecutor()
    executor.env_path = Path("/fake/env")

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    pip_path = executor._get_pip_path()
    assert pip_path == Path("/fake/env/bin/pip")


def test_get_pip_path_windows(monkeypatch):
    executor = PythonExecutor()
    executor.env_path = Path("C:/fake/env")

    monkeypatch.setattr(platform, "system", lambda: "Windows")

    pip_path = executor._get_pip_path()
    assert pip_path == Path("C:/fake/env/Scripts/pip.exe")
