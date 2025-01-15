def run_string_matching(column_list, script_context):
    relevant_columns_list = []
    for column in column_list:
        if column in script_context:
            relevant_columns_list.append(column)
    return relevant_columns_list
