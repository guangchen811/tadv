from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from cadv_exploration.llm._tasks import DVTask
from cadv_exploration.llm.langchain._downstream_task_prompt import ML_INFERENCE_TASK_DESCRIPTION
from cadv_exploration.llm.langchain._prompt import (
    ASSUMPTIONS_EXTRACTION_PROMPT, RELEVANT_COLUMN_TARGET_PROMPT,
    RULE_GENERATION_PROMPT, SYSTEM_TASK_DESCRIPTION)


class LangChainCADV:
    def __init__(self, model: str = None, downstream_task_description: str = ML_INFERENCE_TASK_DESCRIPTION,
                 logger=None):
        if model is None:
            self.model = ChatOpenAI(model="gpt-4o-mini")
        else:
            self.model = ChatOpenAI(model=model)
        self.logger = logger
        self._build_chain(downstream_task_description)

    @staticmethod
    def _build_prompt(task: DVTask,
                      downstream_task_description: str = ML_INFERENCE_TASK_DESCRIPTION) -> ChatPromptTemplate:
        if task == DVTask.RELEVANT_COLUMN_TARGET:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", RELEVANT_COLUMN_TARGET_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )
        elif task == DVTask.EXPECTATION_EXTRACTION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", ASSUMPTIONS_EXTRACTION_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )
        elif task == DVTask.RULE_GENERATION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", RULE_GENERATION_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )

    def _build_single_chain(self, task: DVTask, downstream_task_description: str = ML_INFERENCE_TASK_DESCRIPTION):
        if task == DVTask.RELEVANT_COLUMN_TARGET:
            prompt = self._build_prompt(task, downstream_task_description)
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

    def _build_chain(self, downstream_task_description: str = ML_INFERENCE_TASK_DESCRIPTION):
        self.relevant_column_target_chain = self._build_single_chain(
            DVTask.RELEVANT_COLUMN_TARGET, downstream_task_description
        )
        self.expectation_extraction_chain = self._build_single_chain(
            DVTask.EXPECTATION_EXTRACTION, downstream_task_description
        )
        self.rule_generation_chain = self._build_single_chain(
            DVTask.RULE_GENERATION, downstream_task_description
        )

    def single_invoke(self, input_variables: dict, num_stages: int = 3):
        """
        Args:
            input_variables (dict): Input variables for the pipeline.
            num_stages (int): Number of stages to run in the pipeline.
        """
        relevant_columns_list = self.relevant_column_target_chain.invoke(
            {
                "code_snippet": input_variables["script"],
                "columns_desc": input_variables["column_desc"],
            }
        )
        if num_stages > 1:
            expectations = self.expectation_extraction_chain.invoke(
                {
                    "code_snippet": input_variables["script"],
                    "columns_desc": input_variables["column_desc"],
                    "relevant_columns": str(relevant_columns_list),
                }
            )
        else:
            expectations = None
        if num_stages > 2:
            rules = self.rule_generation_chain.invoke(
                {"assumptions": expectations, "relevant_columns": relevant_columns_list,
                 "code_snippet": input_variables["script"]})
        else:
            rules = None
        return relevant_columns_list, expectations, rules

    def invoke(self, input_variables: dict, num_stages: int = 3, max_retries: int = 3):
        attempt = 0
        while attempt < max_retries:
            try:
                relevant_columns_list, expectations, suggestions = self.single_invoke(
                    input_variables=input_variables, num_stages=num_stages
                )
                break  # Exit the loop if successful
            except OutputParserException as e:
                attempt += 1
                self.logger.error(f"Attempt {attempt} failed with error: {e}")
                if attempt >= max_retries:
                    self.logger.error("All retry attempts failed.")
                    raise e
            except Exception as e:
                self.logger.error("An unexpected error occurred.")
                raise e  # Raise any other unexpected exceptions
        return relevant_columns_list, expectations, suggestions
