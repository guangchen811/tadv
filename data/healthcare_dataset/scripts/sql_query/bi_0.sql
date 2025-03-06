WITH risk_factors AS (
    SELECT
        *,
        CUME_DIST() OVER (PARTITION BY "Medical Condition" ORDER BY "Billing Amount" DESC) AS condition_billing_cume_dist,
        AVG("Billing Amount") OVER (PARTITION BY "Medical Condition") AS avg_billing_per_condition,
        PERCENT_RANK() OVER (PARTITION BY "Insurance Provider" ORDER BY "Billing Amount" DESC) AS insurance_billing_rank,
        ROW_NUMBER() OVER (PARTITION BY "Medical Condition", "Insurance Provider" ORDER BY "Billing Amount" DESC) AS row_num
    FROM new_data
),

high_risk_patients AS (
    SELECT * EXCLUDE ("Room Number", "Doctor", "Date of Admission", "Discharge Date", "Hospital")
    FROM risk_factors
    WHERE
        condition_billing_cume_dist <= 0.10  -- Top 10% of high-billing patients per condition
        OR insurance_billing_rank <= 0.05    -- Top 5% most expensive patients per insurance provider
),

insurance_coverage_issues AS (
    SELECT
        "Insurance Provider",
        COUNT(*) AS num_high_risk_patients,
        AVG("Billing Amount") AS avg_billing,
        COUNT(DISTINCT "Medical Condition") AS distinct_conditions
    FROM high_risk_patients
    GROUP BY "Insurance Provider"
    HAVING COUNT(*) > 500  -- Highlight insurance providers with excessive high-billing patients
)

SELECT * EXCLUDE ("row_num")
FROM high_risk_patients p
LEFT JOIN insurance_coverage_issues i
ON p."Insurance Provider" = i."Insurance Provider"
ORDER BY p."Billing Amount" DESC;