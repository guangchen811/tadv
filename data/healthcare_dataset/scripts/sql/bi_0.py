class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
    SELECT 
    corr("Age", "Billing Amount") AS age_billing_correlation
FROM train;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return []
