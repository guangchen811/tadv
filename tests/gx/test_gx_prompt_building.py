from langchain.prompts import ChatPromptTemplate

from tadv.llm.tasks import DVTask
from tadv.llm.langchain import LangChainTADVGreatExpectationsDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION


def test_gx_prompt_building():
    lang_chain = LangChainTADVGreatExpectationsDialect(model_name="gpt-4o-mini", downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

    task = DVTask.COLUMN_ACCESS_DETECTION
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
