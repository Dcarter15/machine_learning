# machine_learning

# Predicting Business Major’s G.P.A. with Machine Learning
Daniel Z. Carter
Bushnell University 

# Author Note
This is my senior capstone project. My faculty mentor was Ernest Bonat, Ph.D. Computer Engineering Faculty. 

# Abstract
This is a write up detailing the process in which a machine learning model was used on data from Bushnell University to predict G.P.A. of Business Majors based on select features.

# Predicting Business Major’s G.P.A. with Machine Learning
 Research Question: Can a machine learning linear regression model successfully predict a Business Major’s G.P.A. based on credits attempted and credits earned, classes, gender, whether the student is an athlete, style of class (online or in-person), and the grade received for each class?
 
# Preprocessing
The first step was to gather and clean the data. The data that was used for this project was gathered from Bushnell University. The data consists of student data dating back to the 90’s and came in the form of two separate CSV files. In order to begin creating the model the data needed to be cleaned so it can be plugged into the model and not cause an error. The very first step is to import the initial libraries that will be used for the cleanup process.
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

The next step was to read the datasets into PANDAS. 
data = pd.read_csv('DC_Course_List_Clean_2000.csv')
gpa_data = pd.read_csv('DC_Senior_Research_Data (1).csv')

After reading in the two separate CSV files, they needed to be merged into one dataframe. 
complete_data = data.merge(gpa_data, on='ID')

The next step in the preprocessing process was to create a column counting the total credits for each student to use as a feature. This was done with the following code:
count = complete_data.groupby('ID')
count['HRS_EARNED'].count()

df = complete_data.merge(complete_data.groupby('ID')['HRS_EARNED']
           .agg(['sum']), 
         left_on='ID', 
         right_index=True)

The data used in this project was provided by Bushnell University. The credit requirement to graduate is to have at least 124 credits. With this in mind, the data was sorted to only use the students who completed enough credits to graduate from the university. This was done by locating all the data in the total credits column that were greater than or equal to 124:
clean = df.loc[df['sum'] >= 124]

Another feature was created by dividing the total credits column by 4 to take an average amount of credits taken each year for each student. Another feature column was created using the same method as finding the total credits to calculate the total amount of classes that each student took. The last initial steps were to remove any rows that contained empty values and remove any column that were not going to be used as features. 
	The next step in the preprocessing process was to encode the data for the model. This was done using Label Encoder:
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
label encoder categorical columns: four, five, eight
label_encoder_four = LabelEncoder()  
final_df['GENDER'] = label_encoder_four.fit_transform(final_df['GENDER'])
label_encoder_five = LabelEncoder()        
final_df['TRM_CDE'] = label_encoder_five.fit_transform(final_df['TRM_CDE'])
label_encoder_eight= LabelEncoder()        
final_df['Dept'] = label_encoder_eight.fit_transform(final_df['Dept'])
label_encoder_nine= LabelEncoder()        
final_df['Delivery'] = label_encoder_nine.fit_transform(final_df['Delivery'])
label_encoder_six= LabelEncoder()        
final_df['ATH_TEAM_MEMBR'] = label_encoder_six.fit_transform(final_df['ATH_TEAM_MEMBR'])
label_encoder_two= LabelEncoder()        
final_df['Grade'] = label_encoder_two.fit_transform(final_df['Grade'])

After encoding the data a few tests were done to determine if there were any outliers in the data that could produce misleading results. This was done by looking at quartiles and box and whisker plots as well as scatter plots:
 
An example of one of the scatter plots (total credits vs gpa) shows evident outliers in the data. The outliers where then removed to make the model more accurate:
#remove outliers
outlier = final_df.loc[final_df['sum'] <= 300]
outlier = outlier.loc[outlier['count'] <= 90]
outlier = outlier.loc[outlier['career_gpa'] >= 2.5]

The preprocessing phase is almost complete. The last thing that was done was convert the data into the appropriate data types so that the model would accept all of the data and not error out:
outlier["HRS_ATTEMPTED"]=pd.to_numeric(outlier["HRS_ATTEMPTED"],errors='coerce')
outlier = outlier.astype("float64")

# The Model
The first step in creating the model was to call in all of the appropriate libraries and packages in order to create the model:
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

The next step in creating the model was to create a function that will split our data into the training, validation, and the testing sets:
def train_validation_test_split(X, y, test_size=0.2, valid_size=0.5, random_state_initial=50):
    try:
        X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=test_size, random_state=random_state_initial)        
        X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=valid_size, random_state=random_state_initial)               
    except:
        exception_message = sys.exc_info()[0]
        print("An error occurred. {}".format(exception_message))
    return X_train, y_train, X_valid, y_valid, X_test, y_test

This function is very important. What is happening in this block of code is setting up the model for future use. The first line of code under the try is splitting 80% of the data into the training set. The next line under the try is then splitting the remaining 20% in half with 10% going into the validation set to check for overfitting and the last 10% going to the testing split for running through the model for final results. The next step is to split the data into your X (features) and y (target) variable. For this model the gpa was our target and the rest of the columns were used as features. The code for this is shown below:
 X = outlier.drop(labels="career_gpa", axis=1)
 y = outlier["career_gpa"]

The following step is calling the function that was created in order to split the data so that the data is split into the appropriate data splits:
X_train, y_train, X_valid, y_valid, X_test, y_test = train_validation_test_split(X, y, test_size=test_size, random_state_initial=1)

The data is then scaled to fit the model. This is only done on the features because the target is only one value. If y was multiple values y would also need to be scaled to fit. The code for this is:
 scaler = StandardScaler()    
 scaler.fit(X_train)
 X_train = scaler.transform(X_train)
 X_valid = scaler.transform(X_valid)

The next part of the process is defining the model to be used. Many models were tested to determine which was the most accurate. The following are the different models that were tested:
 model_regressor = RandomForestRegressor(n_jobs=-1, random_state=50)
 model_regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
 model_regressor = SVR(kernel='linear', C=100, gamma='auto')
 model_regressor = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
 model_regressor = DecisionTreeRegressor(random_state=0)
 model_regressor = MLPRegressor(random_state=1)
 model_regressor = xgb.XGBRegressor(n_jobs=-1, random_state=1, objective="reg:squarederror")
 model_regressor = LinearRegression()

The next step was to fit the model and run the predictions:

 model_regressor.fit(X_train, y_train)
 y_predicted = model_regressor.predict(X_valid)

The last part of the model is getting the results and calculating the errors. This is done with:

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

Which resulted in these results:


 
# Conclusion
This was a very long and tedious project. The code shown throughout this project is from the final python file that was written for this model. Several attempts were made until the final product was created. The main issue that was found initially was that the data was overfitting to the training set causing the results to be misleading. A cross validation test was conducted as a first attempt to solve this issue. The final resolution was adding in the validation split to negate overfitting. This was not the case for every model as half of the models still overfit the data. This is significant improvement as all of the models were overfitting previously. From the results it is evident that the XGBRegressor model was the most accurate with a tested R^2 value at 0.96. This means that we can expect that roughly 96% of the predictions made from the model are being predicted correctly. 

# References
https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
