from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from cadv_exploration.llm._tasks import DVTask
from cadv_exploration.llm.langchain._prompt import (
    ASSUMPTIONS_EXTRACTION_PROMPT, RELEVANT_COLUMN_TARGET_PROMPT,
    RULE_GENERATION_PROMPT, SYSTEM_TASK_DESCRIPTION)
from cadv_exploration.llm.langchain._downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION


class LangChainCADV:
    def __init__(self, model: str = None):
        if model is None:
            self.model = ChatOpenAI(model="gpt-4o-mini")
        else:
            self.model = ChatOpenAI(model=model)

        self._build_chain()

    def _build_prompt(self, task: DVTask) -> ChatPromptTemplate:
        if task == DVTask.RELEVENT_COLUMN_TARGET:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", RELEVANT_COLUMN_TARGET_PROMPT),
                ],
                partial_variables={"downstream_task_description": ML_INFERENCE_TASK_DESCRIPTION},
            )
        elif task == DVTask.EXPECTATION_EXTRACTION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", ASSUMPTIONS_EXTRACTION_PROMPT),
                ],
                partial_variables={"downstream_task_description": ML_INFERENCE_TASK_DESCRIPTION},
            )
        elif task == DVTask.RULE_GENERATION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", RULE_GENERATION_PROMPT),
                ],
                partial_variables={"downstream_task_description": ML_INFERENCE_TASK_DESCRIPTION},
            )

    def _build_single_chain(self, task: DVTask):
        if task == DVTask.RELEVENT_COLUMN_TARGET:
            prompt = self._build_prompt(task)
            parser = CommaSeparatedListOutputParser()
            single_chain = prompt | self.model | parser
        elif task == DVTask.EXPECTATION_EXTRACTION:
            prompt = self._build_prompt(task)
            parser = JsonOutputParser()
            single_chain = prompt | self.model | parser
        elif task == DVTask.RULE_GENERATION:
            prompt = self._build_prompt(task)
            parser = JsonOutputParser()
            single_chain = prompt | self.model | parser
        else:
            raise ValueError(f"Unknown task {task}")
        return single_chain

    def _build_chain(self):
        self.relevant_column_target_chain = self._build_single_chain(
            DVTask.RELEVENT_COLUMN_TARGET
        )
        self.expectation_extraction_chain = self._build_single_chain(
            DVTask.EXPECTATION_EXTRACTION
        )
        self.rule_generation_chain = self._build_single_chain(DVTask.RULE_GENERATION)

    def invoke(self, input_variables: dict):
        relevant_columns_list = self.relevant_column_target_chain.invoke(
            {
                "code_snippet": input_variables["script"],
                "columns_desc": input_variables["column_desc"],
            }
        )
        expectations = self.expectation_extraction_chain.invoke(
            {
                "code_snippet": input_variables["script"],
                "columns_desc": input_variables["column_desc"],
                "relevant_columns": str(relevant_columns_list),
            }
        )
        rules = self.rule_generation_chain.invoke(
            {"assumptions": expectations, "relevant_columns": relevant_columns_list,
             "code_snippet": input_variables["script"]})
        return relevant_columns_list, expectations, rules
