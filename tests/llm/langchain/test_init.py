from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION


def test_prompt_building():
    chain = LangChainCADV(downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"

    chain = LangChainCADV(model_name="gpt-4o-mini", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"

    chain = LangChainCADV(model_name="llama3.2:1b", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain.model, ChatOllama)
    assert chain.model.model == "llama3.2:1b"

    chain = LangChainCADV(model_name="llama3.2:3b", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain.model, ChatOllama)
    assert chain.model.model == "llama3.2"
