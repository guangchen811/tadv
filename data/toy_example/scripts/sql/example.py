class ColumnDetectionTask:

    def code_type(self):
        return "python"

    @property
    def original_script(self):
        return """
nonsensitive_df = duckdb.sql("SELECT * EXCLUDE ssn, gender, race
FROM 's3://datalake/latest/hospitalisations.csv'").df()
hosp_df = nonsensitive_df.dropna()
strokes_total = duckdb.sql("SELECT COUNT(*) FROM hosp_df
WHERE diagnosis = 'stroke'").fetch()
strokes_for_rare_bloodtypes = duckdb.sql("SELECT COUNT(*)
FROM hosp_df WHERE diagnosis = 'stroke'
AND bloodtype IN ('AB negative', 'B negative')").fetch()
generate_report(strokes_total, strokes_for_rare_bloodtypes)
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ["diagnosis", "bloodtype"]
