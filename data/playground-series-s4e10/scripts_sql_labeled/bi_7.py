class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
-- Analyzing loan data with slightly obscured column references
WITH loan_data AS (
    SELECT
        l.loan_grade AS grade_col,        -- Aliased loan grade
        l.loan_intent AS intent_col,      -- Aliased loan intent
        l.loan_status AS status_col,      -- Aliased loan status
        l.loan_int_rate AS int_rate_col,  -- Aliased loan interest rate
        l.loan_amnt AS amount_col,        -- Aliased loan amount
        l.person_income AS income_col,    -- Aliased borrower income
        l.loan_percent_income AS percent_col -- Aliased loan-to-income percentage
    FROM train l
)
SELECT
    grade_col AS grade_segment,            -- Display segment by loan grade
    intent_col AS intent_segment,          -- Display segment by loan intent
    COUNT(*) AS total_loans,               -- Total number of loans
    SUM(CASE WHEN status_col = 1 THEN 1 ELSE 0 END) AS total_defaults, -- Count defaults
    ROUND(SUM(CASE WHEN status_col = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate, -- Default percentage
    ROUND(AVG(int_rate_col), 2) AS avg_interest_rate, -- Average interest rate
    ROUND(AVG(amount_col), 2) AS avg_loan_amount,     -- Average loan amount
    ROUND(AVG(income_col), 2) AS avg_income,          -- Average borrower income
    ROUND(AVG(percent_col), 2) AS avg_percent_income  -- Average loan percent income
FROM loan_data
GROUP BY grade_col, intent_col
ORDER BY default_rate DESC, grade_segment, intent_segment;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_grade', 'loan_intent', 'loan_status', 'loan_int_rate', 'loan_amnt', 'person_income',
                'loan_percent_income']
