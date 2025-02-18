WITH stay_data AS (
    SELECT
        CAST("Date of Admission" AS DATE) AS admission_date,
        CAST("Discharge Date" AS DATE) AS discharge_date,
        DATEDIFF('day', 
                 CAST("Date of Admission" AS DATE), 
                 CAST("Discharge Date" AS DATE)) AS length_of_stay
    FROM train
)
SELECT
    strftime('%Y-%m', admission_date) AS month_period,
    ROUND(AVG(length_of_stay), 2)      AS avg_length_of_stay,
    COUNT(*)                           AS admission_count
FROM stay_data
GROUP BY 1
ORDER BY 1;