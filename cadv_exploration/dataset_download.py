import os
from dotenv import load_dotenv
import kaggle

# Load environment variables from .env file
load_dotenv()

dataset_name = "prasad22/healthcare-dataset"

kaggle.api.dataset_download_files(
    dataset_name, path=f"./data/{dataset_name}/files", unzip=True
)

res = kaggle.api.kernels_list(dataset=dataset_name, page=1, page_size=20)
for r in res:
    file = kaggle.api.kernels_pull(r.ref, path=f"./data/{dataset_name}/kernels")
    print(f"Pulled {file} to ./data/{dataset_name}/kernels")
