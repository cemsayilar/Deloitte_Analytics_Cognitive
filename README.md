# Question 1 - SQL
## I create this readme file as a illustrative and elegant presentation. I enjoyed to prepare it, I hope you would love it too.
Sincierly
B. Cem Sayilar
## SQL Querys

### a) There are four types of joins in SQL:
```sql
/*INNER JOIN: returns only the matching rows from both tables
/* a) There are four types of joins in SQL:
/*LEFT JOIN: returns all rows from the left table and the matching rows from the right table
/*RIGHT JOIN: returns all rows from the right table and the matching rows from the left table
/*FULL OUTER JOIN: returns all rows from both tables, with NULL values for non-matching rows.
```

### b) SQL to display all the workers of the Company:
```sql
SELECT EMPLOYEES.EMP_ID, EMPLOYEES.FIRST_NAME, EMPLOYEES.LAST_NAME, EMPLOYEES.JOB_ROLE, EMPLOYEES.START_DATE, DEVELOPERS.DEPARTMENT, DEVELOPERS.CONTRACT_TYPE, DEVELOPERS.SALARY
FROM EMPLOYEES
LEFT JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID
```

### c) SQL to query the names and the Start Date of the developers those have FULL_TIME contract:
```sql
SELECT FIRST_NAME, LAST_NAME, START_DATE
FROM EMPLOYEES
JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID
WHERE CONTRACT_TYPE = 'FULL_TIME' AND JOB_ROLE = 'Developer'
```

### d) SQL to display number employees work in each job role:
```sql
SELECT JOB_ROLE, COUNT(*)
FROM EMPLOYEES
GROUP BY JOB_ROLE
```

### e) SQL to display Number of new employees per Department Name by year:
```sql
/* Find my solution and visualization in python_case file.
```

### f) SQL to display the departments whose average salary is equal or higher than 2000$:
```sql
SELECT DEPARTMENT_NAME, AVG(SALARY) AS AVERAGE_SALARY
FROM DEVELOPERS
LEFT JOIN
(SELECT ID, Department_Name FROM DEPARTMENT)
AS DEP ON DEVELOPERS.DEPARTMENT = DEP.ID
GROUP BY DEPARTMENT_NAME
HAVING AVG(SALARY) >= 2000
```

# Question 2 - Python
``` python

## Imports
import numpy as np
import pandas as pd

## Model building
# Tree based
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
# Non-tree based
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate

## Visualization
import numpy as np
import matplotlib.pyplot as plt
################################### Case Data / 1 ##########################################
N = 51
b0 = 0
b1 = 2
x = np.arange(0,N,1)
y = b0 + b1*x
# generate noise and add it, y = b0 + b1*x + e

mn = 0
sd = 5
np.random.seed(100)
e = np.random.normal(mn,sd,N)
y = y + e
################################### Case Data / 1 ##########################################
################################### Case Tasks / 1 ##########################################
# 1. Fit a tree based model (e.g. decision tree, random forest, gradient boosting, etc.)
# 2. Fit a non-tree based model (e.g. SVM, Ridge, Lasso, Elastic Net, etc.)
# 3. Predict x=100 with both models
# 4. Which model is able to predict more correctly? Why? Explain the predictions of both models.
# 5. [OPTIONAL] Visualize the predictions from x=1 to 100 with 1 increment
################################### Case Tasks ##########################################

# Creating Dataframe
case_df = pd.DataFrame(columns= ['x_values', 'y_values'])
case_df['x_values'] = x
case_df['y_values'] = y

# Split the dataframe into training and testing sets
train_df = case_df.iloc[:40, :]
test_df = case_df.iloc[40:, :]

# 1. Fit tree-based model

# Tree-based models
tree_models = [RandomForestRegressor(random_state=100), GradientBoostingRegressor()]
tree_model_names = ['Random Forest', 'Gradient Boosting']

# Non-tree-based models
non_tree_models = [LinearRegression(), Ridge(alpha=1.0), Lasso(), ElasticNet(), SVR()]
non_tree_model_names = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR']

# Fit predict the models for x_new
x_new = [[100]]
predictions = []
for model in tree_models:
    model.fit(train_df[['x_values']], train_df['y_values'])
    predictions.append(model.predict(x_new)[0])

for model in non_tree_models:
    model.fit(train_df[['x_values']], train_df['y_values'])
    predictions.append(model.predict(x_new)[0])

# Print the results
for model_name, pred in zip(tree_model_names+non_tree_model_names, predictions):
    print(f'{model_name}: {pred}')

# 4. Which model is able to predict more correctly? Why? Explain the predictions of both models.
print(f"Random Forest Prediction: {rf_pred}")
print(f"Ridge Prediction: {ridge_pred}")

# Non-tree based linear models outperform tree based models.
# Since I don't have the actual value of y for x=100, cannot say which model predicts more accurately.
# But, ussualy, in a richer and more complex datasets, I excpect higher training performance because
# relationship between x and y in this case is nonlinear, and tree based models are better for capturing nonlinear
# relationships compared to linear models like Ridge, or Linear Regression based model. (LR)
# I can observe this situation in graph.



# 5. [OPTIONAL] Visualize the predictions from x=1 to 100 with 1 increment
import numpy as np
import matplotlib.pyplot as plt

x_plot = np.arange(1, 101, 1)
models = [rf_model,
          gb_model,
          LR_model,
          ridge_model,
          lasso_model,
          elastic_model,
          SVR_model]
model_names = ['Random Forest',
               'Gradient Boosting',
               'Logistic Regression',
               'Ridge',
               'Lasso',
               'ElasticNet',
               'SGDOneClassSVM']

# A for loop to predict and plot for each model
for model, model_name in zip(models, model_names):
    model_plot = model.predict(x_plot.reshape(-1, 1))
    plt.plot(x_plot, model_plot, label=model_name)

# Plot the scatter plot and show the legend
plt.scatter(case_df['x_values'],
            case_df['y_values'],
            color='black')
plt.legend()
plt.show()

```

