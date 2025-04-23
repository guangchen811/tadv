from langchain_core.runnables.base import RunnableSequence

from tadv.llm.langchain import SequentialLangChainTADVDeequDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from tadv.llm.tasks import DVTask
from tadv.utils import load_dotenv


def test_build_single_chain():
    load_dotenv()
    langchain = SequentialLangChainTADVDeequDialect(model_name="gpt-4o-mini",
                                                    downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

    column_access_detection = DVTask.COLUMN_ACCESS_DETECTION
    chain = langchain._build_single_chain(column_access_detection,
                                          downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain, RunnableSequence)

    expectation_extraction_task = DVTask.EXPECTATION_EXTRACTION
    chain = langchain._build_single_chain(expectation_extraction_task,
                                          downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain, RunnableSequence)
