import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

outlier = pd.read_csv('machine_learning.csv')
outlier.head()

outlier = outlier.dropna(axis=0, how='any')
outlier["hours_attempted"] = pd.to_numeric(outlier["hours_attempted"], errors='coerce')
outlier = outlier.astype("float64")
outlier.info()
#outlier.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\encoded_data.csv", index=False)

#result = final_df.apply(lambda x: sum(x.isnull()), axis=0) 
X = outlier.drop(labels="career_gpa", axis=1)
#X = np.array(X)
#print(X)
    
#     define label
y = outlier["career_gpa"]
#y =  np.array(y)
#print(y)

#     data split (20% test and 80% train)
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
   
#     data standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()    
scaler.fit(X_train)
#y_scaler = scaler.fit(y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#X_train_scaled = X_scaler.transform(X_train)
#X_test_scaled = X_scaler.transform(X_test)
#y_train_scaled = y_scaler.transform(y_train)
#y_test_scaled = y_scaler.transform(y_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#model_regressor= RandomForestRegressor(n_jobs=-1, random_state=50)
#model_regressor= SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#model_regressor = DecisionTreeRegressor(random_state=0)
#model_regressor = MLPRegressor(random_state=42)
model_regressor = xgb.XGBRegressor(n_jobs=-1, random_state=5, objective='reg:squarederror')
#model_regressor = LinearRegression()
#model_regressor= SVR(kernel='linear')
#model_regressor= SVR(kernel='poly')
# model_regressor = SVR(kernel = "poly")
print(model_regressor)
print()
#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#svr_lin = SVR(kernel='linear', C=100, gamma='auto')
#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
           #    coef0=1)

model_regressor.fit(X_train, y_train)

y_predicted = model_regressor.predict(X_test)
y_train_predicted = model_regressor.predict(X_train)

# R2 can take values from 0 to 1. A value of 1 indicates that the regression predictions perfectly fit the data. 
r2_train_score_value = r2_score(y_train, y_train_predicted)
print("r2 train score value:")
print(r2_train_score_value)
print()

r2_score_value = r2_score(y_test, y_predicted)
print("r2 score value:")
print(r2_score_value)
print()

# small MSE suggests the model is great at prediction
train_mean_squared_error_result = mean_squared_error(y_train, y_train_predicted)
# mean_squared_error_result = np.sqrt(mean_squared_error_result)
print("train mean squared error:")
print(train_mean_squared_error_result)
print()

mean_squared_error_result = mean_squared_error(y_test, y_predicted)
# mean_squared_error_result = np.sqrt(mean_squared_error_result)
print("mean squared error:")
print(mean_squared_error_result)
print()

# small MAE suggests the model is great at prediction
train_mean_absolute_error_result = mean_absolute_error(y_train, y_train_predicted)
print("train mean absolute error:")
print(train_mean_absolute_error_result)
print()

mean_absolute_error_result = mean_absolute_error(y_test, y_predicted)
print("mean absolute error:")
print(mean_absolute_error_result)
print()

from scipy.stats import sem
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
#https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
def evaluate_model(X, y, repeats):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# create model
	model = model_regressor
	# evaluate model
	scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
	return scores
# configurations to test
repeats = range(1,6)
results = list()
for r in repeats:
	# evaluate using a given number of repeats
	scores = evaluate_model(X, y, r)
	# summarize
	print('>%d mean=%.4f se=%.3f' % (r, np.mean(scores), sem(scores)))
	# store
	results.append(scores)
# plot the results
plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_predicted, y_test, edgecolors=(0, 0, 1))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()

predictions = {'Predictions': y_predicted,
                'Actual': y_test}
predictions_df = pd.DataFrame(predictions, columns = ['Predictions', 'Actual'])
predictions_df