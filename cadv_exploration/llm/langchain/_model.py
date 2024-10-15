import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)


# from cadv_exploration.utils import get_project_root
# from cadv_exploration.utils import load_dotenv

from cadv_exploration.llm.langchain._prompt import (
    SYSTEM_TASK_DESCRIPTION,
    RELEVENT_COLUMN_TARGET_PROMPT,
    EXPECTATION_EXTRACTION_PROMPT,
)

from cadv_exploration.llm._tasks import DVTask


class LangChainCADV:
    def __init__(self, model: str = None):
        if model is None:
            self.llm = ChatOpenAI()
        else:
            self.llm = ChatOpenAI(model=model)

    def build_prompt(self, task: DVTask) -> ChatPromptTemplate:
        if task == DVTask.RELEVENT_COLUMN_TARGET:
            return ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", RELEVENT_COLUMN_TARGET_PROMPT),
                ],
            )
        elif task == DVTask.EXPECTATION_EXTRACTION:
            return ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", EXPECTATION_EXTRACTION_PROMPT),
                ],
            )
        elif task == DVTask.CHECK_FORMULATION:
            raise NotImplementedError("Check formulation task not yet implemented")

    def _build_single_chain(self, task: DVTask):
        if task == DVTask.RELEVENT_COLUMN_TARGET:
            prompt = self.build_prompt(task)
            parser = CommaSeparatedListOutputParser()
            single_chain = prompt | self.llm | parser
        elif task == DVTask.EXPECTATION_EXTRACTION:
            prompt = self.build_prompt(task)
            parser = JsonOutputParser()
            single_chain = prompt | self.llm | parser
        elif task == DVTask.CHECK_FORMULATION:
            raise NotImplementedError("Check formulation task not yet implemented")
        else:
            raise ValueError(f"Unknown task {task}")
        return single_chain
