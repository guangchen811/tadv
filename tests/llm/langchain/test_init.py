from langchain_openai import ChatOpenAI

from tadv.llm.langchain import LangChainTADV_DEEQU
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION


def test_prompt_building():
    chain = LangChainTADV_DEEQU(model_name="gpt-4o-mini", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain.model, ChatOpenAI)
    assert chain.model.model_name == "gpt-4o-mini"
