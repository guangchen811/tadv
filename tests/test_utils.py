import os

from cadv_exploration.utils import get_project_root, load_dotenv


def test_get_project_root():
    assert get_project_root().name == "cadv-exploration"


def test_load_dotenv():
    load_dotenv()
    assert (
            "OPENAI_API_KEY" in os.environ.keys()
    ), "OPENAI_API_KEY not found in environment variables"
    assert (
            "HF_TOKEN" in os.environ.keys()
    ), "HF_TOKEN not found in environment variables"
    assert (
            "SPARK_VERSION" in os.environ.keys()
    ), "SPARK_VERSION not found in environment variables"
