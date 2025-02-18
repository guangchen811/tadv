WITH daily_admissions AS (SELECT CAST("Date of Admission" AS DATE) AS admission_date,
                                 COUNT(*)                          AS admissions_count
                          FROM train
                          GROUP BY 1)
SELECT admission_date,
       admissions_count,
       ROUND(
               AVG(admissions_count) OVER (
            ORDER BY admission_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
               2
       ) AS rolling_7_day_avg
FROM daily_admissions
ORDER BY admission_date ASC;