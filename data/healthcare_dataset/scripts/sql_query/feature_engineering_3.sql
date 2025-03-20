SELECT t."id",
       SUM(CASE WHEN t."Medical Condition" = 'Cancer' THEN 3 ELSE 0 END
           + CASE WHEN t."Medical Condition" = 'Diabetes' THEN 2 ELSE 0 END
           + CASE WHEN t."Medical Condition" = 'Hypertension' THEN 1 ELSE 0 END
       ) AS condition_risk_score
FROM new_data AS t
GROUP BY t."id"
ORDER BY t."id";
-- The current score calculation is based on the following logic:
-- - Cancer: 3 points
-- - Diabetes: 2 points
-- - Hypertension: 1 point
-- The total risk score is the sum of the points for each condition.
-- Maybe we should ask the doctor from relevant hospitals to provide a more accurate risk score calculation.