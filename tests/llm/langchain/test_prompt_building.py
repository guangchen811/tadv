from langchain.prompts import ChatPromptTemplate

from cadv_exploration.llm._tasks import DVTask
from cadv_exploration.llm.langchain import LangChainCADV


def test_prompt_building():
    lang_chain = LangChainCADV()

    task = DVTask.RELEVANT_COLUMN_TARGET
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns_desc"]

    task = DVTask.EXPECTATION_EXTRACTION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns_desc", "relevant_columns"]

    task = DVTask.RULE_GENERATION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["assumptions", "code_snippet", "relevant_columns"]
