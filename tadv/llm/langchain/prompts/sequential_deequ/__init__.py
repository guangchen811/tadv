from tadv.llm.langchain.prompts.sequential_deequ import _prompt_with_deequ
from tadv.llm.langchain.prompts.sequential_deequ import _prompt_with_experience
from tadv.llm.langchain.prompts.sequential_deequ._prompt import (
    COLUMN_ACCESS_DETECTION_PROMPT,
    RULE_GENERATION_PROMPT,
    SYSTEM_TASK_DESCRIPTION,
    ASSUMPTIONS_EXTRACTION_PROMPT as DEFAULT_ASSUMPTIONS_PROMPT,
)


def get_assumptions_prompt(strategy: str = None):
    """
    Return the appropriate ASSUMPTIONS_EXTRACTION_PROMPT based on the selected strategy.

    Parameters:
    - strategy: None, "with_experience", or "with_deequ"

    Returns:
    - Corresponding ASSUMPTIONS_EXTRACTION_PROMPT string.

    Raises:
    - ValueError if strategy is unknown.
    """
    if strategy is None:
        return DEFAULT_ASSUMPTIONS_PROMPT
    elif strategy == "with_experience":
        return _prompt_with_experience.ASSUMPTIONS_EXTRACTION_PROMPT
    elif strategy == "with_deequ":
        return _prompt_with_deequ.ASSUMPTIONS_EXTRACTION_PROMPT
    else:
        raise ValueError(f"Unknown assumption generation trick: {strategy}")
