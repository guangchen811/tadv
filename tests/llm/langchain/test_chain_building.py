from langchain_core.runnables.base import RunnableSequence

from cadv_exploration.llm._tasks import DVTask
from cadv_exploration.llm.langchain import LangChainCADV
from cadv_exploration.llm.langchain.downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from cadv_exploration.utils import load_dotenv


def test_build_single_chain():
    load_dotenv()
    langchain = LangChainCADV(downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)

    relevant_column_target_task = DVTask.RELEVANT_COLUMN_TARGET
    chain = langchain._build_single_chain(relevant_column_target_task,
                                          downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain, RunnableSequence)

    expectation_extraction_task = DVTask.EXPECTATION_EXTRACTION
    chain = langchain._build_single_chain(expectation_extraction_task,
                                          downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    assert isinstance(chain, RunnableSequence)
