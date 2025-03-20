-- Create a Common Table Expression (CTE) to compute daily admission counts
WITH daily_admissions
         AS (SELECT CAST("Date of Admission" AS DATE) AS admission_date,  -- Convert admission date to DATE type
                    COUNT(*)                          AS admissions_count -- Count the number of admissions for each date
             FROM new_data
             GROUP BY 1 -- Group by admission date (column index 1)
    )

-- Main query to compute a 7-day rolling average of admissions
SELECT admission_date,
       admissions_count, -- Daily number of admissions

       -- Compute the rolling 7-day average including the current day
       ROUND(
               AVG(admissions_count) OVER (
            ORDER BY admission_date  -- Order by date to ensure chronological calculation
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW  -- Include the last 6 days + current day
        ),
               2 -- Round the result to 2 decimal places
       ) AS rolling_7_day_avg

FROM daily_admissions
ORDER BY admission_date ASC; -- Sort results in ascending order by date