WITH admission_windows AS (SELECT "id",
                                  CAST("Date of Admission" AS DATE) AS admission_date,
                                  -- Count admissions within a 30-day rolling window
                                  COUNT(*)                             OVER (PARTITION BY "id" ORDER BY CAST("Date of Admission" AS DATE) ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS count_30_day_window
                           FROM new_data)
-- Flag frequent visitors who have more than 1 admission in a 30-day window
SELECT "id",
       admission_date,
       count_30_day_window,
       -- Assign a flag if there is more than one admission within the 30-day period
       CASE WHEN count_30_day_window > 1 THEN 1 ELSE 0 END AS frequent_visitor_flag
FROM admission_windows
ORDER BY "id", admission_date;
-- Future considerations: Exploring patterns in "Medical Condition" and "Admission Type" may provide deeper insights into frequent visits. Examining "Billing Amount" trends within repeated admissions might highlight cost implications. Including "Doctor" and "Hospital" in future queries could help identify whether certain providers are associated with frequent readmissions. "Insurance Provider" data could be analyzed to check if specific policies correlate with multiple hospital visits. Further filtering by "Room Number" might indicate whether certain room types are linked to frequent stays.