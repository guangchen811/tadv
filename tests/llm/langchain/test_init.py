from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from cadv_exploration.llm._tasks import DVTask
from cadv_exploration.llm.langchain._model import LangChainCADV


def test_prompt_building():
    chain = LangChainCADV()
    assert isinstance(chain.llm, ChatOpenAI)
    assert chain.llm.model_name == "gpt-4o-mini"

    chain = LangChainCADV(model="gpt-4o-mini")
    assert isinstance(chain.llm, ChatOpenAI)
    assert chain.llm.model_name == "gpt-4o-mini"
