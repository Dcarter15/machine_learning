import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

final_df = pd.read_csv('clean.csv')
final_df.head()

#remove outliers
outlier = final_df.loc[final_df['sum'] <= 300]
outlier = outlier.loc[outlier['count'] <= 90]
outlier = outlier.loc[outlier['career_gpa'] >= 2.5]
outlier.head()

#final_df = final_df.dropna(axis=0, how='any')
#final_df["HRS_ATTEMPTED"] = pd.to_numeric(final_df["HRS_ATTEMPTED"], errors='coerce')
#final_df = final_df.astype("float64")
#final_df.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\encoded_data.csv", index=False)
outlier = outlier.dropna(axis=0, how='any')
outlier["HRS_ATTEMPTED"] = pd.to_numeric(outlier["HRS_ATTEMPTED"], errors='coerce')
outlier = outlier.astype("float64")
outlier.to_csv(r"C:\Users\15309\OneDrive\Desktop\capstone\encoded_data.csv", index=False)