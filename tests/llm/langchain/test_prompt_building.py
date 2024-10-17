from langchain.prompts import ChatPromptTemplate

from cadv_exploration.llm._tasks import DVTask
from cadv_exploration.llm.langchain._model import LangChainCADV


def test_prompt_building():
    lang_chain = LangChainCADV()

    task = DVTask.RELEVENT_COLUMN_TARGET
    prompt = lang_chain.build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns"]

    task = DVTask.EXPECTATION_EXTRACTION
    prompt = lang_chain.build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns", "relevant_columns"]
