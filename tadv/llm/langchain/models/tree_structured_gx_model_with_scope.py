import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser)
from langchain_core.prompts import ChatPromptTemplate

from tadv.llm.langchain.abstract import TreeStructuredLangChainTADV
from tadv.llm.langchain.llm_backend.entry import get_langchain_model
from tadv.llm.langchain.prompts.sequential_gx_model_with_scope import (COLUMN_ACCESS_DETECTION_PROMPT,
                                                                       CODE_GENERATION_PROMPT, SYSTEM_TASK_DESCRIPTION,
                                                                       DEFAULT_ASSUMPTIONS_PROMPT)
from tadv.llm.langchain.prompts.sequential_gx_model_with_scope import GXConfigManager
from tadv.llm.tasks import SequentialTADVTasks


class TreeStructuredTADVGreatExpectationsDialect(TreeStructuredLangChainTADV):
    def __init__(self, model_name: str = None, downstream_task_description: str = None,
                 assumption_generation_trick: str = None,
                 expectations_text_descriptions_style: str = "Full",
                 logger: logging.Logger = None):
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
    def _build_prompt(task: SequentialTADVTasks,
                      downstream_task_description: str = None,
                      assumption_generation_trick: str = None,
                      expectations_text_descriptions: str = None, ) -> ChatPromptTemplate:
        if task == SequentialTADVTasks.COLUMN_ACCESS_DETECTION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", COLUMN_ACCESS_DETECTION_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )
        elif task == SequentialTADVTasks.ASSUMPTION_EXTRACTION:
            if assumption_generation_trick is None:
                assumptions_extraction_prompt = DEFAULT_ASSUMPTIONS_PROMPT
            elif assumption_generation_trick == "code_with_line_numbers":
                assumptions_extraction_prompt = DEFAULT_ASSUMPTIONS_PROMPT
            elif assumption_generation_trick == "code_with_pygments_highlighting":
                assumptions_extraction_prompt = DEFAULT_ASSUMPTIONS_PROMPT
            else:
                raise ValueError(f"Unknown assumption generation trick: {assumption_generation_trick}")
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", assumptions_extraction_prompt),
                ],
                partial_variables={"downstream_task_description": downstream_task_description},
            )
        elif task == SequentialTADVTasks.CODE_GENERATION:
            return ChatPromptTemplate(
                [
                    ("system", SYSTEM_TASK_DESCRIPTION),
                    ("human", CODE_GENERATION_PROMPT),
                ],
                partial_variables={"downstream_task_description": downstream_task_description,
                                   "expectations_text_descriptions": expectations_text_descriptions},
            )

    def _build_single_chain(self, task: SequentialTADVTasks, downstream_task_description: str = None,
                            assumption_generation_trick: str = None,
                            expectations_text_descriptions_style: str = "Full"):
        if task == SequentialTADVTasks.COLUMN_ACCESS_DETECTION:
            if downstream_task_description is None:
                raise ValueError("Downstream task description is required.")
            prompt = self._build_prompt(task, downstream_task_description=downstream_task_description)
            parser = CommaSeparatedListOutputParser()
            single_chain = prompt | self.model | parser
        elif task == SequentialTADVTasks.ASSUMPTION_EXTRACTION:
            prompt = self._build_prompt(task, downstream_task_description=downstream_task_description,
                                        assumption_generation_trick=assumption_generation_trick)
            parser = JsonOutputParser()
            single_chain = prompt | self.model | parser
        elif task == SequentialTADVTasks.CODE_GENERATION:
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
        self.column_access_detection_chain = self._build_single_chain(
            SequentialTADVTasks.COLUMN_ACCESS_DETECTION, downstream_task_description=downstream_task_description
        )
        self.expectation_extraction_chain = self._build_single_chain(
            SequentialTADVTasks.ASSUMPTION_EXTRACTION, assumption_generation_trick=assumption_generation_trick
        )
        self.rule_generation_chain = self._build_single_chain(
            SequentialTADVTasks.CODE_GENERATION, downstream_task_description, expectations_text_descriptions_style
        )

    def show_prompts(self, input_variables: dict, num_stages: int):
        """
        Args:
            input_variables (dict): Input variables for the pipeline.
            num_stages (int): Number of stages to run in the pipeline.
        """
        pass

    def invoke(self, input_variables: dict, num_stages: int = 3):
        """
        Args:
            input_variables (dict): Input variables for the pipeline.
            num_stages (int): Number of stages to run in the pipeline.
        """
        code_snippet = input_variables["script"]
        if self.assumption_generation_trick == "code_with_line_numbers":
            code_snippet = self._add_line_numbers(code_snippet)
        elif self.assumption_generation_trick == "code_with_pygments_highlighting":
            code_snippet = self._add_pygments_highlighting(code_snippet)
        accessed_columns_list = self.column_access_detection_chain.invoke(
            {
                "code_snippet": code_snippet,
                "columns_desc": input_variables["column_desc"],
            }
        )
        if num_stages > 1:
            if (self.assumption_generation_trick is None
                    or self.assumption_generation_trick == "code_with_line_numbers"
                    or self.assumption_generation_trick == "code_with_pygments_highlighting"):
                expectations = self.expectation_extraction_chain.invoke(
                    {
                        "code_snippet": code_snippet,
                        "columns_desc": input_variables["column_desc"],
                        "accessed_columns": str(accessed_columns_list),
                    }
                )
            else:
                raise ValueError(f"Unknown assumption generation trick: {self.assumption_generation_trick}")
        else:
            expectations = None
        if num_stages > 2:
            rules = self.rule_generation_chain.invoke(
                {"assumptions": expectations, "accessed_columns": accessed_columns_list,
                 "code_snippet": code_snippet})
        else:
            rules = None
        return accessed_columns_list, expectations, rules

    def invoke_with_retries(self, input_variables: dict, num_stages: int = 3, max_retries: int = 3):
        accessed_columns_list, expectations, suggestions = None, None, None
        attempt = 0
        while attempt < max_retries:
            try:
                accessed_columns_list, expectations, suggestions = self.invoke(input_variables=input_variables,
                                                                               num_stages=num_stages)
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

    @staticmethod
    def _add_line_numbers(code_snippet: str) -> str:
        """
        Adds line numbers to the code snippet.
        Args:
            code_snippet (str): The code snippet to add line numbers to.
        Returns:
            str: The code snippet with line numbers added.
        """
        num_lines = len(code_snippet.strip().split('\n'))
        if num_lines > 10000:
            raise ValueError("Code snippet has more than 10000 lines.")

        code_snippet_with_line_numbers = "\n".join(
            f"{i:04}: {line}" for i, line in enumerate(code_snippet.strip().split('\n'), start=1)
        )

        return code_snippet_with_line_numbers

    @staticmethod
    def _add_pygments_highlighting(code_snippet: str) -> str:
        """
        Adds Pygments highlighting to the code snippet.
        Args:
            code_snippet (str): The code snippet to add Pygments highlighting to.
        Returns:
            str: The code snippet with Pygments highlighting added.
        """
        from pygments import highlight
        from pygments.lexers import guess_lexer
        from pygments.formatters import TerminalFormatter
        from pygments.util import ClassNotFound
        try:
            lexer = guess_lexer(code_snippet)
        except ClassNotFound:
            raise ValueError("Could not guess lexer for the code snippet.")
        formatter = TerminalFormatter(linenos=True)
        highlighted_code = highlight(code_snippet, lexer, formatter)
        return highlighted_code
