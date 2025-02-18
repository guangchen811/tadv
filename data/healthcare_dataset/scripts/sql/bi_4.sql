SELECT
    "Insurance Provider",
    "Medical Condition",
    ROUND(AVG("Billing Amount"), 2) AS avg_billing
FROM train
GROUP BY GROUPING SETS (
    ("Insurance Provider", "Medical Condition"),
    ("Insurance Provider"),
    ("Medical Condition")
)
ORDER BY "Insurance Provider", "Medical Condition";