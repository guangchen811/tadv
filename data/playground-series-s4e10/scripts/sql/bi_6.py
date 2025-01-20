class ColumnDetectionTask:

    @property
    def original_code(self):
        return """
SELECT train.loan_grade,
       train.loan_intent,
       COUNT(*)                                                                            AS total_loans,
       SUM(CASE WHEN train.loan_status = 1 THEN 1 ELSE 0 END)                              AS total_defaults,
       ROUND(SUM(CASE WHEN train.loan_status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate_percentage,
       ROUND(AVG(train.loan_amnt), 2)                                                      AS avg_loan_amount,
       ROUND(AVG(train.person_income), 2)                                                  AS avg_borrower_income,
       ROUND(AVG(train.loan_percent_income), 2)                                            AS avg_loan_percent_income
FROM train
GROUP BY loan_grade,
         loan_intent
ORDER BY default_rate_percentage DESC,
         loan_grade,
         loan_intent;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['loan_grade', 'loan_intent', 'loan_amnt']
