import os
from urllib.parse import urlparse

# List of URLs (truncated here for brevity, but assumed to be complete in real input)
urls = [
    "https://greatexpectations.io/expectations/expect_column_mean_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_min_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_median_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_most_common_value_to_be_in_set/",
    "https://greatexpectations.io/expectations/expect_column_pair_values_a_to_be_greater_than_b/",
    "https://greatexpectations.io/expectations/expect_column_pair_values_to_be_equal/",
    "https://greatexpectations.io/expectations/expect_column_pair_values_to_be_in_set/",
    "https://greatexpectations.io/expectations/expect_column_quantile_values_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_proportion_of_unique_values_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_stdev_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_sum_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_to_exist/",
    "https://greatexpectations.io/expectations/expect_column_unique_value_count_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_value_lengths_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_value_lengths_to_equal/",
    "https://greatexpectations.io/expectations/expect_column_value_z_scores_to_be_less_than/",
    "https://greatexpectations.io/expectations/expect_column_values_to_be_between/",
    "https://greatexpectations.io/expectations/expect_column_values_to_be_in_set/",
    "https://greatexpectations.io/expectations/expect_column_values_to_be_in_type_list/",
    "https://greatexpectations.io/expectations/expect_column_values_to_be_null/",
    "https://greatexpectations.io/expectations/expect_column_values_to_be_of_type/",
    "https://greatexpectations.io/expectations/expect_column_values_to_be_unique/",
    "https://greatexpectations.io/expectations/expect_column_values_to_match_like_pattern/",
    "https://greatexpectations.io/expectations/expect_column_values_to_match_like_pattern_list/",
    "https://greatexpectations.io/expectations/expect_column_values_to_match_regex/",
    "https://greatexpectations.io/expectations/expect_column_values_to_match_regex_list/",
    "https://greatexpectations.io/expectations/expect_column_values_to_not_be_in_set/",
    "https://greatexpectations.io/expectations/expect_column_values_to_not_be_null/",
    "https://greatexpectations.io/expectations/expect_column_values_to_not_match_like_pattern/",
    "https://greatexpectations.io/expectations/expect_column_values_to_not_match_like_pattern_list/",
    "https://greatexpectations.io/expectations/expect_column_values_to_not_match_regex/",
    "https://greatexpectations.io/expectations/expect_column_values_to_not_match_regex_list/",
    "https://greatexpectations.io/expectations/expect_compound_columns_to_be_unique/",
    "https://greatexpectations.io/expectations/expect_multicolumn_sum_to_equal/",
    "https://greatexpectations.io/expectations/expect_select_column_values_to_be_unique_within_record/",
    "https://greatexpectations.io/expectations/expect_table_column_count_to_be_between/",
    "https://greatexpectations.io/expectations/expect_table_column_count_to_equal/",
    "https://greatexpectations.io/expectations/expect_table_columns_to_match_ordered_list/",
    "https://greatexpectations.io/expectations/expect_table_columns_to_match_set/",
    "https://greatexpectations.io/expectations/expect_table_row_count_to_be_between/",
    "https://greatexpectations.io/expectations/expect_table_row_count_to_equal/",
    "https://greatexpectations.io/expectations/expect_table_row_count_to_equal_other_table/",
    "https://greatexpectations.io/expectations/unexpected_rows_expectation/",
]

# Generate dictionary of filename to URL
yaml_data = {}
for url in urls:
    path = urlparse(url).path
    slug = os.path.basename(path.strip("/"))
    if slug:
        yaml_data[f"{slug}.yaml"] = {"url": url}

import yaml
from pathlib import Path

# Create YAML files for each entry
output_dir = Path("./Expectations")
output_dir.mkdir(parents=True, exist_ok=True)

for filename, content in yaml_data.items():
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        yaml.dump(content, f)

list(output_dir.iterdir())
