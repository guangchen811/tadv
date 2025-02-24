SELECT t."id",
       SUM(CASE WHEN t."Medical Condition" = 'Cancer' THEN 3 ELSE 0 END
           + CASE WHEN t."Medical Condition" = 'Diabetes' THEN 2 ELSE 0 END
           + CASE WHEN t."Medical Condition" = 'Hypertension' THEN 1 ELSE 0 END
       ) AS condition_risk_score
FROM new_data AS t
GROUP BY t."id"
ORDER BY t."id";