import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('DC_Course_List_Clean_2000.csv')
gpa_data = pd.read_csv('DC_Senior_Research_Data (1).csv')
gpa_data.head()

data.head()

student_id = data.groupby('ID')
student_id.count()

complete_data = data.merge(gpa_data, on='ID')
complete_data.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\uoclass.csv", index=False)
print(complete_data['ID'].nunique())
print(complete_data['ID'].count())

count = complete_data.groupby('ID')
count['HRS_EARNED'].count()

df = complete_data.merge(complete_data.groupby('ID')['HRS_EARNED']
           .agg(['sum']), 
         left_on='ID', 
         right_index=True)


clean = df.loc[df['sum'] >= 124]
clean.head()

clean['Average_CreditsperYear'] = clean['sum']/4
clean.head()

#course_length = len(clean['CRS_TITLE'])
#students = len(clean['ID'].unique())
#average_course = course_length/students
#average_course
#class_count = clean.groupby('ID')['CRS_TITLE'].agg(['count'])
#class_count
df = clean.merge(clean.groupby('ID')['CRS_TITLE']
           .agg(['count']), 
         left_on='ID', 
         right_index=True)
df.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\clean.csv", index=False)

print(df['career_gpa'].nunique())
print(df['ID'].nunique())

final_df = df.drop(columns=['ID','YR_CDE','CourseID','Unnamed: 12','ENTRY_DTE','EXIT_DTE','EXIT_REASON','DEGREE_DESC','MAJOR_MINOR_DESC', 'CRS_TITLE'])
final_df
final_df = final_df.dropna()
final_df.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\clean.csv", index=False)
final_df.head()

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# label encoder categorical columns: four, five, eight
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
final_df.head()

#find and outliers
quartiles = final_df['sum'].quantile([.25,.5,.75])
lowerq = quartiles[0.25]
upperq = quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of occupancy is: {lowerq}")
print(f"The upper quartile of occupancy is: {upperq}")
print(f"The interquartile range of occupancy is: {iqr}")
print(f"The median of occupancy is: {quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")

#outlier_occupancy = final_df.loc[(final_df['sum'] < 
#        lower_bound) | (final_df['sum'] > upper_bound)]
#outlier_occupancy

#find and outliers
quartiles = final_df['sum'].quantile([.25,.5,.75])
lowerq = quartiles[0.25]
upperq = quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of occupancy is: {lowerq}")
print(f"The upper quartile of occupancy is: {upperq}")
print(f"The interquartile range of occupancy is: {iqr}")
print(f"The median of occupancy is: {quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")

#outlier_occupancy = final_df.loc[(final_df['sum'] < 
#        lower_bound) | (final_df['sum'] > upper_bound)]
#outlier_occupancy

fig1, ax1 = plt.subplots()
ax1.boxplot(final_df['sum'])
plt.show()
fig1, ax1 = plt.subplots()
ax1.boxplot(final_df['count'])
plt.show()
plt.scatter(final_df['count'], final_df['career_gpa'])
plt.show()
plt.scatter(final_df['sum'], final_df['career_gpa'])
plt.show()

final_df.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\clean.csv", index=False)
