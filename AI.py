import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ds = pd.read_csv('C:/Users/alber/Documents/GitHub/274481/credit_prediction.csv') #open and read the file using pandas

#ds = pd.read_csv("C:/Users/matte/Desktop/274481/credit_prediction.csv")

#Evaluation
ds.head() #show the first elements
correlation = ds.corr() #to see correlations, but all categorical ones will be ignored, so change them

from sklearn.preprocessing import LabelEncoder
ds['Credit_Mix'] = LabelEncoder().fit_transform(ds['Credit_Mix'])
ds['Payment_Behaviour'] = LabelEncoder().fit_transform(ds['Payment_Behaviour'])
#ds['Credit_Score'] = LabelEncoder().fit_transform(ds['Credit_Score'])

ds.head()
#again
ds.corr()
description = ds.describe() #statistical summary
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

ds = ds.drop(columns=['id'])
#collect data into arrays
X = ds.iloc[:, :-1].values      
y = ds.iloc[:, -1].values

#remove outliers


from numpy import percentile
q25, q75 = percentile(ds['Annual_Income'], 25), percentile(ds['Annual_Income'], 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))


cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

outliers = [x for x in ds['Annual_Income'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

outliers_removed = [x for x in ds['Annual_Income'] if x > lower and x < upper]
print('Non-outlier observations: %d' % len(outliers_removed))

annual_income_mean = ds['Annual_Income'].mean()



for i in outliers:
    ds['Annual_Income'] = ds['Annual_Income'].replace(i,annual_income_mean)


'''
from scipy import stats

df =pd.DataFrame({'Age':[int(i) for i in ds['Age']]})
df['z_score']=stats.zscore(df['Age'])
df_no = df.loc[df['z_score'].abs()<2.1060025]


df1 =pd.DataFrame({'Num_Bank_Accounts':[int(i) for i in ds['Num_Bank_Accounts']]})
df1['z_score']=stats.zscore(df1['Num_Bank_Accounts'])
df1.loc[df1['z_score'].abs()<3.0]

df2 =pd.DataFrame({'Monthly_Inhand_Salary':[int(i) for i in ds['Monthly_Inhand_Salary']]})
df2['z_score']=stats.zscore(df2['Monthly_Inhand_Salary'])
df2.loc[df1['z_score'].abs()<3.0]
'''

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
model = "b"

if model == "SVC":
    #classifier = SVC(kernel = 'linear', random_state = 0)
    param_grid = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.1,1, 10, 100]}
    classifier = SVC()
    clf = GridSearchCV(classifier, param_grid)
    clf.fit(X_train,y_train)
elif model=="C":
    classifier = XGBClassifier(use_label_encoder=False) #using XGboost as more efficient 
else :
    k_range = list(range(5, 7))
    param_grid = dict(n_neighbors=k_range)
    knn = neighbors.KNeighborsClassifier()
    # defining parameter range
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
    # fitting the model for grid search
    grid_search=grid.fit(X_train, y_train)
    #param_grid = {'n_neighbors':[3,5,7], 'weights':[0.1,1, 10, 100]}
    #classifier = SVC()
    #clf = GridSearchCV(classifier, param_grid)
    #clf.fit(X_train,y_train)
    #classifier = neighbors.KNeighborsClassifier(n_neighbors = 3, weights='uniform')

'''
y_pred = clf.best_estimator_.predict(X_test) # this contains the best trained classifier
print(classification_report(y_test, y_pred))
'''
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print(scores)




param_grid = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.1,1, 10, 100]}
classifier = SVC()
clf = GridSearchCV(classifier, param_grid)
clf.fit(X_train,y_train)
y_pred = clf.best_estimator_.predict(X_test) # this contains the best trained classifier
print(classification_report(y_test, y_pred))