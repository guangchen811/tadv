-- Categorizing patients into age buckets for analysis
SELECT "id",                 -- Unique identifier for each patient
       "Age",                -- Patient's age, used to determine the bucket classification

       CASE
           WHEN "Age" < 18 THEN 'child' -- Patients under 18 categorized as children
           WHEN "Age" BETWEEN 18 AND 30 THEN 'young_adult' -- Young adults between 18 and 30
           WHEN "Age" BETWEEN 31 AND 50 THEN 'adult' -- Adults aged 31 to 50
           WHEN "Age" BETWEEN 51 AND 64 THEN 'mid_senior' -- Middle-aged to senior group (51 to 64)
           ELSE 'senior' -- Patients 65 and older categorized as seniors
           END AS age_bucket -- Assigning an age category to each patient

FROM new_data
ORDER BY "id";
-- Sorting results by patient ID for consistency

/*
Thought process:
- The age groups were chosen based on general classifications, but it’s best to confirm with a doctor.
- Are these ranges clinically meaningful? Should "senior" be further divided (e.g., 65-80 vs. 80+)?
- If this data is used for medical purposes, should we align it with standard medical age brackets (e.g., pediatric, geriatric)?
- Leaving the ranges flexible for adjustment based on expert input.
*/