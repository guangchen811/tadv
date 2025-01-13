from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from cadv_exploration.llm.langchain import LangChainCADV


def test_prompt_building():
    chain = LangChainCADV()
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"

    chain = LangChainCADV(model_name="gpt-4o-mini")
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"

    chain = LangChainCADV(model_name="llama3.2:1b")
    assert isinstance(chain.model, ChatOllama)
    assert chain.model.model == "llama3.2:1b"

    chain = LangChainCADV(model_name="llama3.2:3b")
    assert isinstance(chain.model, ChatOllama)
    assert chain.model.model == "llama3.2:3b"
