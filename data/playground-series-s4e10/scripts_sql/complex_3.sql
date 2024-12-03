WITH CategoricalAnalysis AS (
    -- Step 1: Aggregate data for each categorical column
    SELECT
        'person_home_ownership' AS category_type,
        person_home_ownership AS category_value,
        COUNT(*) AS total_count,
        AVG(loan_amnt) AS avg_loan_amount,
        AVG(loan_int_rate) AS avg_interest_rate,
        SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
    FROM test
    GROUP BY person_home_ownership

    UNION ALL

    SELECT
        'loan_intent' AS category_type,
        loan_intent AS category_value,
        COUNT(*) AS total_count,
        AVG(loan_amnt) AS avg_loan_amount,
        AVG(loan_int_rate) AS avg_interest_rate,
        SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
    FROM test
    GROUP BY loan_intent

    UNION ALL

    SELECT
        'loan_grade' AS category_type,
        loan_grade AS category_value,
        COUNT(*) AS total_count,
        AVG(loan_amnt) AS avg_loan_amount,
        AVG(loan_int_rate) AS avg_interest_rate,
        SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
    FROM test
    GROUP BY loan_grade

    UNION ALL

    SELECT
        'cb_person_default_on_file' AS category_type,
        cb_person_default_on_file AS category_value,
        COUNT(*) AS total_count,
        AVG(loan_amnt) AS avg_loan_amount,
        AVG(loan_int_rate) AS avg_interest_rate,
        SUM(CASE WHEN cb_person_default_on_file = 'Y' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS default_rate
    FROM test
    GROUP BY cb_person_default_on_file
)
-- Step 2: Rank categories based on default rates
SELECT
    category_type,
    category_value,
    total_count,
    avg_loan_amount,
    avg_interest_rate,
    default_rate,
    RANK() OVER (PARTITION BY category_type ORDER BY default_rate DESC) AS default_rate_rank
FROM CategoricalAnalysis
ORDER BY category_type, default_rate_rank;