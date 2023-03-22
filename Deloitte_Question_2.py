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
import xgboost
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

# Fit predict the models for 100
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


