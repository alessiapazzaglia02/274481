# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:36:57 2022

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ds = pd.read_csv("./credit_prediction.csv") #open and read the file using pandas
#Evaluation
ds.head() #show the first elements
ds.corr() #to see correlations, but all categorical ones will be ignored, so change them
from sklearn.preprocessing import LabelEncoder
ds['Credit_Mix'] = LabelEncoder().fit_transform(ds['Credit_Mix'])
ds['Payment_Behaviour'] = LabelEncoder().fit_transform(ds['Payment_Behaviour'])
ds.head()
#again
ds.corr()
ds.describe() #statistical summary
ds.info() 
ds.shape
print(pd.value_counts(ds['Credit_Score']))
_ = plt.hist(ds['Credit_Score']) # Assign the result of plt.hist to variable _. This is a common trick to discard unwanted output.
plt.show()

#pre-process
#we can eliminate a priori some columns, as we can consider them as not relevant 
ds = ds.drop(columns=['Occupation', 'Type_of_Loan', 'Credit_History_Age', 'Changed_Credit_Limit', 'Payment_of_Min_Amount'])
#impute missing values
ds = ds.replace(np.nan,00)
#df=df.fillna('NM') --> forse ha senso rimpiazzare valori con media (o mediana) usando df.mean(), quindi df=df.fillna(df.mean()), per non perdere quei dati che andranno poi elaborati e avere una stringa quando bisogna fare i conti è scomodo
#df=df.replace(['No Data'], 'NM') --> forse ha senso rimpiazzare valori con media (o mediana) usando df.mean(), quindi df=df.replace(['No Data'], df.mean()), per non perdere quei dati che andranno poi elaborati e avere una stringa quando bisogna fare i conti è scomodo


#collect data into arrays
X = ds.iloc[:, :-1].values      
y = ds.iloc[:, -1].values

#remove outliers
from scipy import stats

df =pd.DataFrame({'Age':[int(i) for i in ds['Age']]})
df['z_score']=stats.zscore(df['Age'])
df_no = df.loc[df['z_score'].abs()<=3]


df1 =pd.DataFrame({'Num_Bank_Accounts':[int(i) for i in ds['Num_Bank_Accounts']]})
df1['z_score']=stats.zscore(df1['Num_Bank_Accounts'])
df1.loc[df1['z_score'].abs()<=3]

df2 =pd.DataFrame({'Monthly_Inhand_Salary':[int(i) for i in ds['Monthly_Inhand_Salary']]})
df2['z_score']=stats.zscore(df2['Monthly_Inhand_Salary'])
df2.loc[df1['z_score'].abs()<=3]


#start the splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#encode data. This step is really important, we have already transformed the categorical data into integers. REMEMBER, NEVER TRAIN ON THE TEST SET!!
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#validation from the training
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from xgboost import XGBClassifier
model = "SVC"

if model == "SVC":
    classifier = SVC(kernel = 'linear', random_state = 0)
elif model=="C":
    classifier = XGBClassifier() #using XGboost as more efficient 
else :
    classifier = neighbors.KNeighborsClassifier(n_neighbors = 3, weights='uniform')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print(scores)