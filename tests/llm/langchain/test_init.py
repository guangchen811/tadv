from langchain_openai import ChatOpenAI

from tadv.llm.langchain import LangChainTADV
from tadv.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION


def test_prompt_building():
    chain = LangChainTADV(model_name="gpt-4o-mini", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"
