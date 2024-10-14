from cadv_exploration.utils import load_dotenv
from langchain.prompts import ChatPromptTemplate

from cadv_exploration.llm.langchain._model import LangChainCADV
from cadv_exploration.llm._tasks import DVTask
from langchain_core.runnables.base import RunnableSequence


def test_build_single_chain():
    load_dotenv()
    langchain = LangChainCADV()

    relevent_column_target_task = DVTask.RELEVENT_COLUMN_TARGET
    chain = langchain._build_single_chain(relevent_column_target_task)
    assert isinstance(chain, RunnableSequence)

    expectation_extraction_task = DVTask.EXPECTATION_EXTRACTION
    chain = langchain._build_single_chain(expectation_extraction_task)
    assert isinstance(chain, RunnableSequence)


def test_build_whole_chain():
    langchain = LangChainCADV()
    chain = langchain.build_whole_chain()
    assert isinstance(chain, RunnableSequence)
