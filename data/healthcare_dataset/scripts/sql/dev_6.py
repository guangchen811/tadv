class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT
    "id",
    "Name",
    "Gender",
    "Insurance Provider",
    "Admission Type",
    "Medical Condition"
FROM train
WHERE ( :insurance_provider_filter IS NULL 
        OR "Insurance Provider" = :insurance_provider_filter )
  AND ( :gender_filter IS NULL 
        OR "Gender" = :gender_filter )
  AND ( :admission_type_filter IS NULL
        OR "Admission Type" = :admission_type_filter )
ORDER BY "id" ASC
LIMIT 100;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
