import os
from cadv_exploration.utils import get_project_root
import dotenv

dotenv.load_dotenv(get_project_root() / ".env")

from langchain_openai import ChatOpenAI


class LangChainCADV:
    def __init__(self):
        self.llm = ChatOpenAI()
