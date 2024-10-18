import os
import subprocess


class KaggleExecutor:
    def run(self, local_data_path, script_path, is_competition):
        # Run the Kaggle environment in Docker
        if is_competition:
            docker_data_path = "/kaggle/input/"
        else:
            docker_data_path = f"/kaggle/input/{os.path.basename(os.path.abspath(local_data_path))}/"
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.join(os.path.abspath(local_data_path), 'files/')}:{docker_data_path}",
            "-v",
            f"{os.path.abspath(script_path)}:/kaggle/script.py",
            "kaggle-env/python:1.0.0",
            "python",
            "/kaggle/script.py",
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Execution successful!")
            print(result.stdout.decode())
        else:
            print("Execution failed:")
            print(result.stderr.decode())

        return result
