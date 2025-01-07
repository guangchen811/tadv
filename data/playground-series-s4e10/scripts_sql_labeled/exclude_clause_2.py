class KaggleLoanColumnDetectionTask:

    @property
    def original_code(self):
        return """
WITH CategoricalColumns AS (
    -- Step 1: Select only categorical columns using EXCLUDE
    SELECT * EXCLUDE (
        person_age,
        person_income,
        person_emp_length,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length
    )
    FROM test),
     AggregatedData AS (
         -- Step 2: Aggregate statistics for each categorical column
         SELECT 'person_home_ownership'                                                           AS category_type,
                person_home_ownership                                                             AS category_value,
                COUNT(*)                                                                          AS total_count,
                SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
         FROM CategoricalColumns
         GROUP BY person_home_ownership

         UNION ALL

         SELECT 'loan_intent'                                                                     AS category_type,
                loan_intent                                                                       AS category_value,
                COUNT(*)                                                                          AS total_count,
                SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
         FROM CategoricalColumns
         GROUP BY loan_intent

         UNION ALL

         SELECT 'loan_grade'                                                                      AS category_type,
                loan_grade                                                                        AS category_value,
                COUNT(*)                                                                          AS total_count,
                SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
         FROM CategoricalColumns
         GROUP BY loan_grade

         UNION ALL

         SELECT 'cb_person_default_on_file'                                                       AS category_type,
                cb_person_default_on_file                                                         AS category_value,
                COUNT(*)                                                                          AS total_count,
                SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
         FROM CategoricalColumns
         GROUP BY cb_person_default_on_file)
-- Step 3: Rank categories based on default rates
SELECT category_type,
       category_value,
       total_count,
       default_rate,
       RANK() OVER (PARTITION BY category_type ORDER BY default_rate DESC) AS default_rate_rank
FROM AggregatedData
ORDER BY category_type, default_rate_rank;
"""

    def required_columns(self):
        # Ground truth for columns used in the ML pipeline
        return ['cb_person_default_on_file', 'loan_grade',
                'loan_intent', 'person_home_ownership',
                ]
