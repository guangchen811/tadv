import os

import kaggle
from dotenv import load_dotenv
from nbconvert import PythonExporter

from cadv_exploration.utils import get_project_root


def download_dataset(dataset_owner, dataset_name):
    load_dotenv()
    data_root_path = get_project_root() / "data"
    dataset_full_name = f"{dataset_owner}/{dataset_name}"
    kaggle.api.dataset_download_files(
        dataset_name, path=data_root_path / dataset_full_name / "files", unzip=True
    )

    res = kaggle.api.kernels_list(dataset=dataset_name, page=1, page_size=3)
    export = PythonExporter()
    for r in res:
        file = kaggle.api.kernels_pull(
            r.ref, path=f"./data/{dataset_name}/kernels_ipynb"
        )
        print(f"Pulled {file} to ./data/{dataset_name}/kernels")
        res = export.from_filename(file)
        text = res[0]
        os.makedirs(f"./data/{dataset_name}/kernels_py", exist_ok=True)
        with open(
                f"./data/{dataset_name}/kernels_py/{file.split('/')[-1].replace('.ipynb', '.py')}",
                "w",
        ) as f:
            f.write(text)
        print(f"Converted {file} to ./data/{dataset_name}/kernels_py")


def download_competition(competition_name, page_size=10, sort_by=None):
    data_root_path = get_project_root() / "data"
    kaggle.api.competition_download_files(
        competition=competition_name,
        path=data_root_path / competition_name / "files",
    )
    res = kaggle.api.kernels_list(competition=competition_name, page=1, page_size=page_size, sort_by=sort_by)
    export = PythonExporter()
    for r in res:
        file = kaggle.api.kernels_pull(
            r.ref, path=data_root_path / competition_name / "kernels_ipynb"
        )
        try:
            res = export.from_filename(file)
            text = res[0]
            os.makedirs(data_root_path / competition_name / "kernels_py", exist_ok=True)

            with open(
                    f"{data_root_path.absolute()}/{competition_name}/kernels_py/{file.split('/')[-1].replace('.ipynb', '.py')}",
                    "w",
            ) as f:
                f.write(text)
        except Exception as e:
            print(f"Error converting {file} to python: {e}")
        print(f"Converted {file} to ./data/{competition_name}/kernels_py")
