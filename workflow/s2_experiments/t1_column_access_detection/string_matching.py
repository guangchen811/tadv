from tadv.llm.langchain import LangChainTADVDeequDialect
from tadv.llm.langchain.prompts.downstream_task_prompt import SQL_QUERY_TASK_DESCRIPTION, ML_INFERENCE_TASK_DESCRIPTION, \
    WEB_TASK_DESCRIPTION


def run_string_matching_for_rcd(column_list, script_context):
    script_context = script_context.lower()
    accessed_columns_list = []
    for column in column_list:
        column_variations = [column,
                             column.replace("_", " "),
                             column.replace("_", ""),
                             column.replace(" ", "_"),
                             column.replace(" ", "")
                             ]
        column_variations_lower = [variation.lower() for variation in column_variations]
        if any([variation in script_context for variation in column_variations_lower]):
            accessed_columns_list.append(column)
    return accessed_columns_list


def run_llm_for_rcd(column_desc, model_name, script_context, task_group):
    input_variables = {
        "column_desc": column_desc,
        "script": script_context,
    }
    if task_group == 'sql_query':
        lc = LangChainTADVDeequDialect(model_name=model_name, downstream_task_description=SQL_QUERY_TASK_DESCRIPTION)
    elif task_group == 'ml_inference':
        lc = LangChainTADVDeequDialect(model_name=model_name, downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    elif task_group == 'webpage_generation':
        lc = LangChainTADVDeequDialect(model_name=model_name, downstream_task_description=WEB_TASK_DESCRIPTION)
    else:
        raise ValueError(f"Unknown task group: {task_group}")
    max_retries = 3
    accessed_columns_list, expectations, suggestions = lc.invoke(
        input_variables=input_variables, num_stages=1, max_retries=max_retries
    )
    accessed_columns_list = sorted(accessed_columns_list, key=lambda x: x.lower())
    return accessed_columns_list
