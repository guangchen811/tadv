from langchain.prompts import ChatPromptTemplate

from tadv.llm._tasks import DVTask
from tadv.llm.langchain import LangChainCADV
from tadv.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION


def test_prompt_building():
    lang_chain = LangChainCADV(downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

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
