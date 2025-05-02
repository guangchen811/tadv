from langchain.prompts import ChatPromptTemplate

from tadv.llm.langchain.models.sequential_gx_model import SequentialLangChainTADVGreatExpectationsDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from tadv.llm.tasks import SequentialTADVTasks


def test_gx_prompt_building():
    lang_chain = SequentialLangChainTADVGreatExpectationsDialect(model_name="gpt-4o-mini",
                                                                 downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

    task = SequentialTADVTasks.COLUMN_ACCESS_DETECTION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.input_variables == ["code_snippet", "columns_desc"]

    task = SequentialTADVTasks.ASSUMPTION_EXTRACTION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert sorted(prompt.input_variables) == sorted(["code_snippet", "columns_desc", "accessed_columns"])

    task = SequentialTADVTasks.CODE_GENERATION
    prompt = lang_chain._build_prompt(task)
    assert isinstance(prompt, ChatPromptTemplate)
    assert sorted(prompt.input_variables) == sorted(["assumptions", "code_snippet", "accessed_columns"])
