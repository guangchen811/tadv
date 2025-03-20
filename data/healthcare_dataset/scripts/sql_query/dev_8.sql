SELECT "id",
       "Name",
       "Age",
       "Medical Condition",
       CASE
           WHEN "Age" >= 65 OR "Medical Condition" IN ('Cancer', 'Diabetes') THEN 'High'
           ELSE 'Low/Medium' END AS risk_level
FROM new_data
WHERE ("Age" >= 65 OR "Medical Condition" IN ('Cancer', 'Diabetes'))
ORDER BY "Age" DESC, "Medical Condition";
-- TODO: ask for clarification on the definition of "High" risk patients from the doctor.
-- Future steps: Consider integrating "Test Results" and "Medication" usage for a more refined risk assessment. Exploring "Billing Amount" and "Insurance Provider" may help identify cost implications for high-risk patients. Further filtering based on "Hospital" and "Doctor" could reveal variations in risk management approaches. "Room Number" and "Admission Type" might indicate whether high-risk patients require special accommodations.