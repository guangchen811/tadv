WITH admission_windows AS (SELECT "id",
                                  CAST("Date of Admission" AS DATE) AS admission_date,

                                  -- Count admissions within a 30-day rolling window
                                  COUNT(*)                             OVER (
            PARTITION BY "id"
            ORDER BY CAST("Date of Admission" AS DATE)
            ROWS BETWEEN 30 PRECEDING AND CURRENT ROW  -- Fixed: Changed RANGE to ROWS
        ) AS count_30_day_window

                           FROM new_data)

-- Flag frequent visitors who have more than 1 admission in a 30-day window
SELECT "id",
       admission_date,
       count_30_day_window,

       -- Assign a flag if there is more than one admission within the 30-day period
       CASE
           WHEN count_30_day_window > 1 THEN 1
           ELSE 0
           END AS frequent_visitor_flag

FROM admission_windows
ORDER BY "id", admission_date;