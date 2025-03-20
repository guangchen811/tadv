from tadv.llm.langchain import LangChainCADV
from tadv.llm.langchain.downstream_task_prompt import SQL_QUERY_TASK_DESCRIPTION, ML_INFERENCE_TASK_DESCRIPTION, \
    WEB_TASK_DESCRIPTION


def run_string_matching_for_rcd(column_list, script_context):
    script_context = script_context.lower()
    relevant_columns_list = []
    for column in column_list:
        column_variations = [column,
                             column.replace("_", " "),
                             column.replace("_", ""),
                             column.replace(" ", "_"),
                             column.replace(" ", "")
                             ]
        column_variations_lower = [variation.lower() for variation in column_variations]
        if any([variation in script_context for variation in column_variations_lower]):
            relevant_columns_list.append(column)
    return relevant_columns_list


def run_llm_for_rcd(column_desc, model_name, script_context, task_group):
    input_variables = {
        "column_desc": column_desc,
        "script": script_context,
    }
    if task_group == 'sql_query':
        lc = LangChainCADV(model_name=model_name, downstream_task_description=SQL_QUERY_TASK_DESCRIPTION)
    elif task_group == 'ml_inference':
        lc = LangChainCADV(model_name=model_name, downstream_task_description=ML_INFERENCE_TASK_DESCRIPTION)
    elif task_group == 'webpage_generation':
        lc = LangChainCADV(model_name=model_name, downstream_task_description=WEB_TASK_DESCRIPTION)
    else:
        raise ValueError(f"Unknown task group: {task_group}")
    max_retries = 3
    relevant_columns_list, expectations, suggestions = lc.invoke(
        input_variables=input_variables, num_stages=1, max_retries=max_retries
    )
    relevant_columns_list = sorted(relevant_columns_list, key=lambda x: x.lower())
    return relevant_columns_list
