/* a)
/*There are four types of joins in SQL:

/*INNER JOIN: returns only the matching rows from both tables
/*LEFT JOIN: returns all rows from the left table and the matching rows from the right table
/*RIGHT JOIN: returns all rows from the right table and the matching rows from the left table
/*FULL OUTER JOIN: returns all rows from both tables, with NULL values for non-matching rows.

/* b) SQL to display all the workers of the Company:

SELECT EMPLOYEES.EMP_ID, EMPLOYEES.FIRST_NAME, EMPLOYEES.LAST_NAME, EMPLOYEES.JOB_ROLE, EMPLOYEES.START_DATE, DEVELOPERS.DEPARTMENT, DEVELOPERS.CONTRACT_TYPE, DEVELOPERS.SALARY
FROM EMPLOYEES
LEFT JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID
