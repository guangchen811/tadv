SELECT "id",
       "Name",
       "Age",
       "Medical Condition",
       CASE
           WHEN "Age" >= 65 OR "Medical Condition" IN ('Cancer', 'Diabetes') THEN 'High'
           ELSE 'Low/Medium'
           END AS risk_level
FROM new_data
-- Return only the "High" risk patients
WHERE (
          "Age" >= 65
              OR "Medical Condition" IN ('Cancer', 'Diabetes')
          )
ORDER BY "Age" DESC, "Medical Condition";
-- TODO: ask for clarification on the definition of "High" risk patients from the doctor