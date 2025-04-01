from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from tadv.runtime_environments import PythonExecutor


@pytest.mark.parametrize("env_exists, env_up_to_date, should_create, should_update", [
    (False, False, True, True),  # Env doesn't exist → create and update
    (False, True, True, False),  # Env doesn't exist but will be up to date after creation
    (True, False, False, True),  # Env exists but outdated → update only
    (True, True, False, False),  # Env exists and up to date → do nothing
])
def test_create_or_update_environment_behavior(tmp_path, env_exists, env_up_to_date, should_create, should_update):
    executor = PythonExecutor()

    # Point to a fake env path in tmp
    executor.env_path = tmp_path / "fake_env"
    executor.env_path.mkdir()

    # Simulate whether pyvenv.cfg exists
    pyvenv_cfg = executor.env_path / "pyvenv.cfg"
    if env_exists:
        pyvenv_cfg.write_text("version = 3.11")

    # Mock internal methods
    executor._create_environment = MagicMock()
    executor._check_env_against_requirements = MagicMock(return_value=env_up_to_date)
    executor._update_environment = MagicMock()

    executor._create_or_update_environment()

    if should_create:
        executor._create_environment.assert_called_once()
    else:
        executor._create_environment.assert_not_called()

    if should_update:
        executor._update_environment.assert_called_once()
    else:
        executor._update_environment.assert_not_called()


def test_update_environment_calls_pip_correctly():
    executor = PythonExecutor()

    # Mock paths
    fake_pip = Path("/fake/env/bin/pip")
    executor._get_pip_path = MagicMock(return_value=fake_pip)
    executor.requirements_path = Path("/fake/project/requirements.txt")

    with patch("subprocess.check_call") as mock_check_call:
        executor._update_environment()

        # First call: upgrade pip
        mock_check_call.assert_any_call([str(fake_pip), "install", "--upgrade", "pip"])

        # Second call: install requirements
        mock_check_call.assert_any_call([str(fake_pip), "install", "-r", str(executor.requirements_path)])

        # Exactly two calls
        assert mock_check_call.call_count == 2


def test_check_env_against_empty_requirements(tmp_path):
    executor = PythonExecutor()

    # Create a fake requirements.txt with only whitespace or nothing
    requirements_txt = tmp_path / "requirements.txt"
    requirements_txt.write_text("\n   \n")
    executor.requirements_path = requirements_txt

    # Mock pip path and subprocess to ensure they're not called
    executor._get_pip_path = MagicMock()

    # Run the check
    result = executor._check_env_against_requirements()

    # Validate the early-return behavior
    assert result is True
    executor._get_pip_path.assert_not_called()
