from langchain.prompts import ChatPromptTemplate

from tadv.llm.langchain import SequentialLangChainTADVDeequDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from tadv.llm.tasks import SequentialTADVTasks


def test_prompt_building():
    lang_chain = SequentialLangChainTADVDeequDialect(model_name="gpt-4o-mini",
                                                     downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

    task = SequentialTADVTasks.COLUMN_ACCESS_DETECTION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns_desc"]

    task = SequentialTADVTasks.ASSUMPTION_EXTRACTION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns_desc", "accessed_columns"]

    task = SequentialTADVTasks.CODE_GENERATION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["assumptions", "code_snippet", "accessed_columns"]
