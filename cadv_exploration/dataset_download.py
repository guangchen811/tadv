import os

import kaggle
from dotenv import load_dotenv
from nbconvert import PythonExporter


def download_dataset(dataset_owner, dataset_name):
    load_dotenv()
    dataset_full_name = f"{dataset_owner}/{dataset_name}"
    kaggle.api.dataset_download_files(
        dataset_name, path=f"./data/{dataset_name}/files", unzip=True
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
