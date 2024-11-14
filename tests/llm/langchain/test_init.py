from langchain_openai import ChatOpenAI

from cadv_exploration.llm.langchain import LangChainCADV


def test_prompt_building():
    chain = LangChainCADV()
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"

    chain = LangChainCADV(model="gpt-4o-mini")
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"
