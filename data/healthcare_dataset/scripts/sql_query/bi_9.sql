-- Create a Common Table Expression (CTE) to calculate length of hospital stays
WITH stays AS (
    SELECT 
        CAST("Date of Admission" AS DATE) AS admission_date,  -- Convert admission date to DATE type
        CAST("Discharge Date" AS DATE) AS discharge_date,     -- Convert discharge date to DATE type
        
        -- Calculate length of stay in days
        DATEDIFF('day', 
                 CAST("Date of Admission" AS DATE), 
                 CAST("Discharge Date" AS DATE)) AS length_of_stay,  
                 
        "Test Results"  -- Keep test results for grouping later
    FROM new_data
)

-- Main query to analyze length of stay based on test results
SELECT 
    "Test Results",
    ROUND(AVG(length_of_stay), 2) AS avg_stay_days,  -- Compute the average length of stay, rounded to 2 decimal places
    COUNT(*) AS patient_count  -- Count the number of patients for each test result category
FROM stays
GROUP BY "Test Results"  -- Aggregate by test results
ORDER BY avg_stay_days DESC;  -- Sort by longest average stay first