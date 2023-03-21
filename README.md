# Case - SQL

## SQL Querys
```sql
* a)
/*There are four types of joins in SQL:
/*INNER JOIN: returns only the matching rows from both tables
/* a) There are four types of joins in SQL:
/*LEFT JOIN: returns all rows from the left table and the matching rows from the right table
/*RIGHT JOIN: returns all rows from the right table and the matching rows from the left table
/*FULL OUTER JOIN: returns all rows from both tables, with NULL values for non-matching rows.

/* b) SQL to display all the workers of the Company:
SELECT EMPLOYEES.EMP_ID, EMPLOYEES.FIRST_NAME, EMPLOYEES.LAST_NAME, EMPLOYEES.JOB_ROLE, EMPLOYEES.START_DATE, DEVELOPERS.DEPARTMENT, DEVELOPERS.CONTRACT_TYPE, DEVELOPERS.SALARY
FROM EMPLOYEES
LEFT JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID

/* c) SQL to query the names and the Start Date of the developers those have FULL_TIME contract:
SELECT FIRST_NAME, LAST_NAME, START_DATE
FROM EMPLOYEES
JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID
WHERE CONTRACT_TYPE = 'FULL_TIME' AND JOB_ROLE = 'Developer'

/* d) SQL to display number employees work in each job role:
SELECT JOB_ROLE, COUNT(*)
FROM EMPLOYEES
GROUP BY JOB_ROLE

/* e) SQL to display Number of new employees per Department Name by year:
/* Find my solution and visualization in python_case file.

/* f) SQL to display the departments whose average salary is equal or higher than 2000$:
SELECT DEPARTMENT_NAME, AVG(SALARY) AS AVERAGE_SALARY
FROM DEVELOPERS
LEFT JOIN
(SELECT ID, Department_Name FROM DEPARTMENT)
AS DEP ON DEVELOPERS.DEPARTMENT = DEP.ID
GROUP BY DEPARTMENT_NAME
HAVING AVG(SALARY) >= 2000


```
