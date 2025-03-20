-- Binning employment length into categories while excluding personal details
-- Excluding person_age to maintain focus on employment analysis CASE
SELECT new_data.* EXCLUDE person_age,
                                                                                                            WHEN person_emp_length < 2
                                                                                                                THEN 'Junior' -- Employees with less than 2 years of experience
                                                                                                            WHEN person_emp_length BETWEEN 2 AND 5
                                                                                                                THEN 'Mid-level' -- Employees with 2 to 5 years of experience
                                                                                                            WHEN person_emp_length > 5
                                                                                                                THEN 'Senior' -- Employees with more than 5 years of experience
                                                                                                            ELSE 'Unknown' -- Handling potential NULL or missing values
END
AS emp_length_category

FROM new_data;

/*
Decided to exclude person_age since it’s not directly relevant to employment categorization.
If needed, age could be analyzed separately to see how employment length correlates with different life stages.
This query keeps the focus on job experience while avoiding unnecessary personal details.
*/