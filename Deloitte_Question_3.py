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


case_df = pd.DataFrame(X) # I struggled to turn this array to dataframe. It turned out so easy.
case_df['target'] = y


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


import xgboost as xgb
from sklearn.metrics import roc_auc_score



model_xgb = XGBClassifier(scale_pos_weight=(1/0.01), random_state=42)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
print("Confusion Matrix (Cost-Sensitive Learning):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (Cost-Sensitive Learning):\n", classification_report(y_test, y_pred))


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
