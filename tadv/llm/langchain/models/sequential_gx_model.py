import importlib

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser)
from langchain_core.prompts import ChatPromptTemplate

from tadv.llm.langchain.abstract import SequentialLangChainTADV
from tadv.llm.langchain.llm_backend import get_langchain_model
from tadv.llm.langchain.prompts.sequential_gx._manager import GXConfigManager
from tadv.llm.langchain.prompts.sequential_gx._prompt import (COLUMN_ACCESS_DETECTION_PROMPT,
                                                              RULE_GENERATION_PROMPT, SYSTEM_TASK_DESCRIPTION)
from tadv.llm.tasks import DVTask


class SequentialLangChainTADVGreatExpectationsDialect(SequentialLangChainTADV):
    def __init__(self, model_name: str = None, downstream_task_description: str = None,
                 assumption_generation_trick: str = None,
                 expectations_text_descriptions_style: str = "Full",
                 logger: object = None):
        if model_name is None:
            raise ValueError("Model name is required.")
        else:
            self.model = self._get_langchain_model(model_name)
        self.downstream_task_description = downstream_task_description
        self.assumption_generation_trick = assumption_generation_trick
        self.logger = logger
        gx_config_manager = GXConfigManager()
        self.expectations_text_descriptions = gx_config_manager.get_all_text_descriptions()
        self._build_chain(downstream_task_description, assumption_generation_trick,
                          expectations_text_descriptions_style)

    @staticmethod
    def _get_langchain_model(model_name: str):
        return get_langchain_model(model_name)

    @staticmethod
    def _build_prompt(task: DVTask,
                      downstream_task_description: str = None,
                      assumption_generation_trick: str = None,
                      expectations_text_descriptions: str = None, ) -> ChatPromptTemplate:
        if task == DVTask.COLUMN_ACCESS_DETECTION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", COLUMN_ACCESS_DETECTION_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )
        elif task == DVTask.EXPECTATION_EXTRACTION:
            if assumption_generation_trick is None:
                assumptions_extraction_prompt = importlib.import_module(
                    "tadv.llm.langchain.prompts.deequ._prompt"
                ).ASSUMPTIONS_EXTRACTION_PROMPT
            else:
                raise ValueError(f"Unknown assumption generation trick: {assumption_generation_trick}")
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", assumptions_extraction_prompt),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )
        elif task == DVTask.RULE_GENERATION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", RULE_GENERATION_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description,
                                   "expectations_text_descriptions": expectations_text_descriptions},
            )

    def _build_single_chain(self, task: DVTask, downstream_task_description: str = None,
                            assumption_generation_trick: str = None,
                            expectations_text_descriptions_style: str = "Full"):
        if task == DVTask.COLUMN_ACCESS_DETECTION:
            if downstream_task_description is None:
                raise ValueError("Downstream task description is required.")
            prompt = self._build_prompt(task, downstream_task_description=downstream_task_description)
            parser = CommaSeparatedListOutputParser()
            single_chain = prompt | self.model | parser
        elif task == DVTask.EXPECTATION_EXTRACTION:
            prompt = self._build_prompt(task, downstream_task_description=downstream_task_description,
                                        assumption_generation_trick=assumption_generation_trick)
            parser = JsonOutputParser()
            single_chain = prompt | self.model | parser
        elif task == DVTask.RULE_GENERATION:
            if expectations_text_descriptions_style == "Full":
                expectations_text_descriptions = self.expectations_text_descriptions
            else:
                raise ValueError(
                    f"Unknown expectations text descriptions style: {expectations_text_descriptions_style}")
            prompt = self._build_prompt(task, downstream_task_description=downstream_task_description,
                                        assumption_generation_trick=assumption_generation_trick,
                                        expectations_text_descriptions=expectations_text_descriptions)
            parser = JsonOutputParser()
            single_chain = prompt | self.model | parser
        else:
            raise ValueError(f"Unknown task {task}")
        return single_chain

    def _build_chain(self, downstream_task_description: str = None, assumption_generation_trick: str = None
                     , expectations_text_descriptions_style: str = "Full"):
        self.relevant_column_target_chain = self._build_single_chain(
            DVTask.COLUMN_ACCESS_DETECTION, downstream_task_description=downstream_task_description
        )
        self.expectation_extraction_chain = self._build_single_chain(
            DVTask.EXPECTATION_EXTRACTION, assumption_generation_trick=assumption_generation_trick
        )
        self.rule_generation_chain = self._build_single_chain(
            DVTask.RULE_GENERATION, downstream_task_description, expectations_text_descriptions_style
        )

    def single_invoke(self, input_variables: dict, num_stages: int = 3):
        """
        Args:
            input_variables (dict): Input variables for the pipeline.
            num_stages (int): Number of stages to run in the pipeline.
        """
        accessed_columns_list = self.relevant_column_target_chain.invoke(
            {
                "code_snippet": input_variables["script"],
                "columns_desc": input_variables["column_desc"],
            }
        )
        if num_stages > 1:
            if self.assumption_generation_trick == "with_experience" or self.assumption_generation_trick is None:
                expectations = self.expectation_extraction_chain.invoke(
                    {
                        "code_snippet": input_variables["script"],
                        "columns_desc": input_variables["column_desc"],
                        "relevant_columns": str(accessed_columns_list),
                    }
                )
            elif self.assumption_generation_trick == "with_deequ":
                expectations = self.expectation_extraction_chain.invoke(
                    {
                        "code_snippet": input_variables["script"],
                        "columns_desc": input_variables["column_desc"],
                        "relevant_columns": accessed_columns_list,
                        "deequ_assumptions": input_variables["deequ_assumptions"],
                    }
                )
            else:
                raise ValueError(f"Unknown assumption generation trick: {self.assumption_generation_trick}")
        else:
            expectations = None
        if num_stages > 2:
            rules = self.rule_generation_chain.invoke(
                {"assumptions": expectations, "relevant_columns": accessed_columns_list,
                 "code_snippet": input_variables["script"]})
        else:
            rules = None
        return accessed_columns_list, expectations, rules

    def invoke(self, input_variables: dict, num_stages: int = 3, max_retries: int = 3):
        accessed_columns_list, expectations, suggestions = None, None, None
        attempt = 0
        while attempt < max_retries:
            try:
                accessed_columns_list, expectations, suggestions = self.single_invoke(
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
                self.logger.error(f"Error details: {e}")
                raise e  # Raise any other unexpected exceptions
        return accessed_columns_list, expectations, suggestions
