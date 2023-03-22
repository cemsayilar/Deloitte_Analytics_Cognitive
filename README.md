## I create this readme file as a illustrative and elegant presentation. You can find simple versions of each code in repository.
## I enjoyed to prepare it, I hope you would love it too.
Sincerely yours,
B. Cem Sayilar



# Question 1 - SQL Querys
### a) There are four types of joins in SQL:
```sql
/*INNER JOIN: returns only the matching rows from both tables
/* a) There are four types of joins in SQL:
/*LEFT JOIN: returns all rows from the left table and the matching rows from the right table
/*RIGHT JOIN: returns all rows from the right table and the matching rows from the left table
/*FULL OUTER JOIN: returns all rows from both tables, with NULL values for non-matching rows.
```

### b) SQL query to display all the workers of the Company:
```sql
SELECT EMPLOYEES.EMP_ID, EMPLOYEES.FIRST_NAME, EMPLOYEES.LAST_NAME, EMPLOYEES.JOB_ROLE, EMPLOYEES.START_DATE, DEVELOPERS.DEPARTMENT, DEVELOPERS.CONTRACT_TYPE, DEVELOPERS.SALARY
FROM EMPLOYEES
LEFT JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID
```

### c) SQL query for names and the start dates of the developers those have full time contract:
```sql
SELECT FIRST_NAME, LAST_NAME, START_DATE
FROM EMPLOYEES
JOIN DEVELOPERS
ON EMPLOYEES.EMP_ID = DEVELOPERS.EMP_ID
WHERE CONTRACT_TYPE = 'FULL_TIME' AND JOB_ROLE = 'Developer'
```

### d) SQL query to display number employees work in each job role:
```sql
SELECT JOB_ROLE, COUNT(*)
FROM EMPLOYEES
GROUP BY JOB_ROLE
```

### e) SQL query to display number of new employees per department name by year:
```sql
/* Find my solution and visualization in python_case file.
```

### f) SQL query to display the departments those average salary equal or higher then 2000$:
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
################################### Question 1 / Data ##########################################
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
################################### Question 1 / Data ##########################################
################################### Question 1 / Tasks ##########################################
# 1. Fit a tree based model (e.g. decision tree, random forest, gradient boosting, etc.)
# 2. Fit a non-tree based model (e.g. SVM, Ridge, Lasso, Elastic Net, etc.)
# 3. Predict x=100 with both models
# 4. Which model is able to predict more correctly? Why? Explain the predictions of both models.
# 5. [OPTIONAL] Visualize the predictions from x=1 to 100 with 1 increment
################################### Question 1 / Tasks ##########################################

# Creating Dataframe
case_df = pd.DataFrame(columns= ['x_values', 'y_values'])
case_df['x_values'] = x
case_df['y_values'] = y

# Split the dataframe into training and testing sets
train_df = case_df.iloc[:40, :]
test_df = case_df.iloc[40:, :]
```


### 1. Fit tree-based model

``` python
# Tree-based models
tree_models = [RandomForestRegressor(random_state=100), GradientBoostingRegressor()]
tree_model_names = ['Random Forest', 'Gradient Boosting']
```


### 2. Fit non-tree-based models

``` python
non_tree_models = [LinearRegression(), Ridge(alpha=1.0), Lasso(), ElasticNet(), SVR()]
non_tree_model_names = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR']
```


### 3. Fit predict the models for 100

``` python
x_new = [[100]]
predictions = []
for model in tree_models:
    model.fit(train_df[['x_values']], train_df['y_values'])
    predictions.append(model.predict(x_new)[0])

for model in non_tree_models:
    model.fit(train_df[['x_values']], train_df['y_values'])
    predictions.append(model.predict(x_new)[0])
```


### 4. Which model is able to predict more correctly? Why? Explain the predictions of both models.
``` python
for model_name, pred in zip(tree_model_names+non_tree_model_names, predictions):
    print(f'{model_name}: {pred}')
```


Non-tree based linear models outperform tree based models. Since I don't have the actual value of y for x=100, cannot say which model predicts more accurately. But, ussualy, in a richer and more complex datasets, I excpect higher training performance because relationship between x and y in this case is nonlinear, and tree based models are better for capturing nonlinear relationships compared to linear models like Ridge, or Linear Regression based model. (LR) I can observe this situation in graph.



### 5. [OPTIONAL] Visualize the predictions from x=1 to 100 with 1 increment
``` python
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

plt.scatter(case_df['x_values'],
            case_df['y_values'],
            color='black')
plt.legend()
plt.show()

```
### matplot lib plot in question 5
![image](https://user-images.githubusercontent.com/96774646/226745272-24a92f78-39b9-432c-beb4-46c89432c8d3.png)









# Question 3 - Pyhton
``` python
## Imports
import numpy as np
import pandas as pd
# Pre-processing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# Model selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
# Model building
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


################################### Question Data / 2 ###########################################
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, weights=(0.99, 0.01), random_state=42)
################################### Question Data / 2 ###########################################
################################### Question Tasks / 2 ##########################################
# The following classification data is provided for a banking fraud analytics application.
# 1. Fit a classifier
# 2. Choose an appropriate loss function & metric for the problem
# 3. Tune hyperparameters to optimize the selected metric
# 4. Explain the results
################################### Question Tasks ##############################################
```


### Explanetory Data Analysis, in an unconventional way.
### Before I start, make_classification function has some parameters that I should know;
```python
#     n_samples=100 How many samples will be in the dataset? In my case, problem says 1000.
#     n_features=20, It's default, that means I have 20 lists in X.
#     n_informative=2, It's complicated. But it effects how some features created.
#     n_redundant=2, It says how many of the features will be totally random.
#     n_repeated=0, The number of duplicated features
#     n_classes=2, How many labels do we have. I have 2, luckily.
#     n_clusters_per_class=2, The number of clusters per class. It sounds nice...
#     weights=None, The proportions of samples assigned to each class. A true fraud det. classic.
#     flip_y=0.01, Noise setting.
#     class_sep=1.0, Cluster size, higher, easier.
#     hypercube=True, ıdk
#     shift=0.0, Shifting features.
#     scale=1.0, Multiply features by the specified value.
#     shuffle=True, Shuffle the samples and the features.
#     random_state=None We all know.


case_df = pd.DataFrame(X) # I struggled to turn this array to dataframe. It turned out so easy.
case_df['target'] = y
```


### Data is highly inbalanced. To solve this, I can make undersampling or oversampling.
### Then, maybe I can apply PCA to see if it's helps or not.
### I will use SMOTE, an oversampling method used in statistics. I choose SMOTE (oversample) over
### the undersampling because when minorty class has contains valuable info and the difference between
### samples huge, like my case, undersampling can lead to a serious info loss.
```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling using SMOTE
oversampler = SMOTE(random_state=42)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)

# Undersampling using RUS
undersampler = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = undersampler.fit_resample(X_train, y_train)

# Logistic Regression model
model = LogisticRegression(random_state=42)

# Fit model on original data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix (Original Data):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (Original Data):\n", classification_report(y_test, y_pred))

# Fit model on SMOTE data
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)
print("Confusion Matrix (SMOTE Data):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (SMOTE Data):\n", classification_report(y_test, y_pred))

# Fit model on RUS data
model.fit(X_train_rus, y_train_rus)
y_pred = model.predict(X_test)
print("Confusion Matrix (RUS Data):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (RUS Data):\n", classification_report(y_test, y_pred))
```


## Cost-sensitive learning using XGBoost
### By setting scale_pos_weight to the inverse of the class imbalance ratio, I essentially telling the algorithm
### to pay more attention to the minority class during training and to try to reduce the number of
### false negatives. This can result in improved performance on the minority class, which is often
### the goal in imbalanced classification problems.
```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score
```


## Choosing the right loss function for XGBoost Classifier
### Because ROC AUC is calculated based on TPR and FPR, it is less likely to be biased towards the majority class than
### other metrics like accuracy or precision. This is because these metrics do not take into account the imbalance in the
### dataset, and may give the appearance of high performance if the majority class is well-classified while
### the minority class is poorly classified.
### So, I will build model with auc and f1 score as loss functions.
```python
model_xgb = XGBClassifier(scale_pos_weight=(1/0.01), random_state=42)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
print("Confusion Matrix (Cost-Sensitive Learning):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (Cost-Sensitive Learning):\n", classification_report(y_test, y_pred))
```


 I used 3 different built-in objective function (loss function) under the 'objective' parameter in XGBoost:
 1- binary:logistic
 2- binary:logitraw
 3- binary:hinge
 And used 'eval_metric' parameter to measure performance of these functions as 'auc'
 For use 'auc' as metric, objective function must be set as 'binary:logistic'.
 To measure the performance other two loss function, I used 'rmsle' (root mean square log error)
 as 'eval_metric'.

### rmsle as metric, binary:logistic reg as loss function - before hyperparameter opt: 0.6012
### rmsle as metric, binary:logistic reg as loss function - before hyperparameter opt: 0.7195

### roc_auc as metric, binary:logistic reg as loss function - before hyperparameter opt: 0.6012
### roc_auc as metric, binary:logistic reg as loss function - before hyperparameter opt: 0.7195
```python

def hyp_op(X, y, model_name, cv=3, scoring="roc_auc"):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    print("Hyperparameter Optimization....")
    best_model = {}
    if model_name == "cart":
        print(f"########## Decision Tree (CART) ##########")
        classifier = DecisionTreeClassifier()
        params = {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        }
    elif model_name == "knn":
        print(f"########## K-Nearest Neighbors ##########")
        classifier = KNeighborsClassifier()
        params = {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    elif model_name == "xgboost":
        print(f"########## XGBoost ##########")
        classifier = model_xgb
        params = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
            "objective": ["binary:logistic"],
            'eval_metric': ['auc']
        }
    elif model_name == "logistic_regression":
        print(f"########## Logistic Regression ##########")
        classifier = LogisticRegression()
        params = {
            "penalty": ["l1", "l2"],
            "C": [0.1, 0.5, 1, 5, 10],
            "solver": ["liblinear", "saga"]
        }
    elif model_name == "catboost":
        print(f"########## CatBoost ##########")
        classifier = CatBoostClassifier()
        params = {
            "iterations": [100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [3, 5, 7],
            "l2_leaf_reg": [1, 3, 5, 7]
        }
    elif model_name == "lightgbm":
        print(f"########## LightGBM ##########")
        classifier = LGBMClassifier()
        params = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
            "objective": ["binary"],
            "subsample": [1, 0.5, 0.7],
            "metric": ["auc"]
        }
    elif model_name == "random_forest":
        print(f"########## Random Forest ##########")
        classifier = RandomForestClassifier()
        params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5, 10]
        }


    cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
    print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

    gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
    print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
    print(f"{model_name} best params: {gs_best.best_params_}", end="\n\n")
    return final_model

opt_xgb = hyp_op(X_train, y_train, 'xgboost', cv=5)
opt_LR = hyp_op(X_train, y_train, 'logistic_regression', cv=5)

opt_xgb = opt_xgb.fit(X_train, y_train)
y_pred_xgb = opt_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

opt_lr = opt_LR.fit(X_train, y_train)
y_pred_lr  = opt_LR.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
```


# Extract
 1- I examin the data sets, due to fabricated data and I am sure there is no
    nan values, I just examin outliers and the metrics that I build the dataset on.
    These are give me enough insights on data.

 2- Then first think to do is deal with imbalanced distribution of data. I done this
    with imbalanced library, usin SMOTE method. I look into under and oversampling
    methods and their use cases then decide to apply an oversampling method.

 3- Then I build my ml model; xgboost that its commonly used and have proven performance
    in classification.
    3a- I used a trick to say to the model that minorty class is more important to me,
        which cause to model more robust to the False Negative. This perspective can change according to the project needs.
        I mean; if project requaires to be more sensitive on False Positives, which means miss tread a transaction as a FRAUD
        lead serious problem.
    3b- On the other hand, I want to observe positive (FRAUD transactions) more accurate, which means robust the algorithm
        on False Negative values rather then False Positive outcomes. Like I said, it depends on project goals.
    3c- I used my own function to conduct hyper_parameter tuning
        more smootly that I can determine model names and parameters individually to present outcomes.

# Conclusion
    In my point of view, simple logistic regression can outperform a tree based method, which is did.
    But the goal of this task is to represent my data understanding and model selection capablities, as well as
    my hyper_parameter process knowladge. For further implementations, I can create a pipeline for this project,
    export as .pkl file, make deployment with help of ML tools (such as MLflow) then observe on Airflow or Prefect
    enviroment to consistent improvement process.
```


