
# Regression Example with XGBRegressor in Python
# https://www.datatechnotes.com/2019/06/regression-example-with-xgbregressor-in.html

# XGBRegressor with GridSearchCV
# https://www.kaggle.com/jayatou/xgbregressor-with-gridsearchcv

import sys
import time
import os
os.system("cls")

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import sem
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score

# printing all parameters form sklearn model completely
from sklearn import set_config
set_config(print_changed_only=False)
# or
# model.get_params()

# from sklearn.model_selection import KFold
# from sklearn.model_selection import 
# from sklearn.model_selection import 

def train_validation_test_split(X, y, test_size=0.2, valid_size=0.5, random_state_initial=50):
    try:
        X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=test_size, random_state=random_state_initial)        
        X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=valid_size, random_state=random_state_initial)               
    except:
        exception_message = sys.exc_info()[0]
        print("An error occurred. {}".format(exception_message))
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#def evaluate_model(X, y, repeats, model_regressor):
	# prepare the cross-validation procedure
	#cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# create model
	#model = model_regressor
	# evaluate model
	#scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
	#return scores

def main():

    outlier = pd.read_csv('machine_learning.csv')
    outlier.head()

    outlier = outlier.dropna(axis=0, how='any')
    outlier["hours_attempted"] = pd.to_numeric(outlier["hours_attempted"], errors='coerce')
    outlier = outlier.astype("float64")
    outlier.info()

    #print(outlier)
    #print()

    #result = final_df.apply(lambda x: sum(x.isnull()), axis=0) 
    X = outlier.drop(labels="career_gpa", axis=1)
    #X = np.array(X)
    #print(X)
        
    #     define label
    y = outlier["career_gpa"]
    #y =  np.array(y)
    #print(y)
    

    #     data split (20% test and 80% train)

    test_size = 0.2
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1)    
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_validation_test_split(X, y, test_size=test_size, random_state_initial=1)

    #print(X_train)

    #     data standard scaling
    scaler = StandardScaler()    
    scaler.fit(X_train)
    #y_scaler = scaler.fit(y_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    #X_train_scaled = X_scaler.transform(X_train)
    #X_test_scaled = X_scaler.transform(X_test)
    #y_train_scaled = y_scaler.transform(y_train)
    #y_test_scaled = y_scaler.transform(y_test)


    #model_regressor= RandomForestRegressor(n_jobs=-1, random_state=50)
    #model_regressor= SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    #model_regressor = DecisionTreeRegressor(random_state=0)
    #model_regressor = MLPRegressor(random_state=1)
    model_regressor = xgb.XGBRegressor(n_jobs=-1, random_state=1, objective="reg:squarederror")
    # print(model_regressor.get_params())

    #model_regressor = LinearRegression()
    #model_regressor= SVR(kernel='linear')
    #model_regressor= SVR(kernel='poly')
    # model_regressor = SVR(kernel = "poly")       
    #svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    #svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    #svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               #coef0=1)

# {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'importance_type': 'gain', 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None, 'n_estimators': 100, 'n_jobs': -1, 'nthread': None, 'objective': 'reg:linear', 'random_state': 5, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 1, 'verbosity': 1}

# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear'],
#               'learning_rate': [.03, 0.05, .07], #so called `eta` value
#               'max_depth': [5, 6, 7],
#               'min_child_weight': [4],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [500]}

    model_regressor.fit(X_train, y_train)

    y_predicted = model_regressor.predict(X_valid)

    #print(X_train)
    #print(y_train)
    #print(y_predicted)
    #print(X_valid)

    # R2 can take values from 0 to 1. A value of 1 indicates that the regression predictions perfectly fit the data. 
    r2_score_value = r2_score(y_valid, y_predicted)
    print("valid r2 score value:")
    print(r2_score_value)
    print()

    # small MSE suggests the model is great at prediction
    mean_squared_error_result = mean_squared_error(y_valid, y_predicted)
    # mean_squared_error_result = np.sqrt(mean_squared_error_result)
    print("valid mean squared error:")
    print(mean_squared_error_result)
    print()

    # small MAE suggests the model is great at prediction
    mean_absolute_error_result = mean_absolute_error(y_valid, y_predicted)
    print("valid mean absolute error:")
    print(mean_absolute_error_result)
    print()
    #exit()

    X_test = scaler.transform(X_test)

    y_predicted = model_regressor.predict(X_test)

    # R2 can take values from 0 to 1. A value of 1 indicates that the regression predictions perfectly fit the data. 
    r2_score_value = r2_score(y_test, y_predicted)
    print("test r2 score value:")
    print(r2_score_value)
    print()

    # small MSE suggests the model is great at prediction
    mean_squared_error_result = mean_squared_error(y_test, y_predicted)
    # mean_squared_error_result = np.sqrt(mean_squared_error_result)
    print("test mean squared error:")
    print(mean_squared_error_result)
    print()

    # small MAE suggests the model is great at prediction
    mean_absolute_error_result = mean_absolute_error(y_test, y_predicted)
    print("test mean absolute error:")
    print(mean_absolute_error_result)
    print()
        
    # # configurations to test
    # repeats = range(1,6)
    # results = list()
    # for r in repeats:
    #     # evaluate using a given number of repeats
    #     scores = evaluate_model(X, y, r, model_regressor)
    #     # summarize
    #     print('>%d mean=%.4f se=%.3f' % (r, np.mean(scores), sem(scores)))
    #     # store
    #     results.append(scores)
    # # plot the results
    # plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(y_predicted, y_test, edgecolors=(0, 0, 1))
    # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Actual')
    # plt.show()

    # # ???
    predictions = {'Predictions': y_predicted, 'Actual': y_test}
    predictions_df = pd.DataFrame(predictions, columns = ['Predictions', 'Actual'])
    predictions_df

if __name__ == '__main__':    
    main()   