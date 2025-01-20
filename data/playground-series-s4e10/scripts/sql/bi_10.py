class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
WITH employment_data AS (
    SELECT
        p.person_emp_length AS employment_duration,
        p.loan_grade AS grade_segment,
        p.loan_status AS status,
        p.loan_int_rate AS interest_rate
    FROM train p
)
SELECT
    employment_duration AS emp_length,
    grade_segment,
    COUNT(*) AS total_loans,
    ROUND(AVG(interest_rate), 2) AS avg_interest_rate,
    SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS defaults,
    ROUND(SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_percentage
FROM employment_data
GROUP BY employment_duration, grade_segment
ORDER BY default_percentage DESC, emp_length, grade_segment;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['person_emp_length', 'loan_grade', 'loan_status', 'loan_int_rate']
