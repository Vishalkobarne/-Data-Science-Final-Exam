#!/usr/bin/env python
# coding: utf-8

# In[315]:


#Importing nessesary libraries


# In[316]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print("Libraries installed Successfully......")


# In[317]:


#Importing dataset


# In[318]:


dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/epilepsy.data')
print("Dataset Loaded Successfully...")


# In[319]:


dataset.head()


# In[320]:


dataset.tail()


# In[321]:


#Checking shape of the 


# In[322]:


dataset.shape


# In[323]:


#Full Data Summary


# In[324]:


dataset.info()


# In[325]:


#Statistical Summary of Data


# In[326]:


dataset.describe()


# In[327]:


#Checking the datatypes


# In[328]:



dataset.dtypes


# In[329]:


#Finding out Null values in Each Columns


# In[330]:


dataset.isna().sum()


# In[331]:


#Total number of NaN values in the dataset


# In[332]:


dataset.isna().sum().sum()


# In[333]:


#Dropping of Columns having significant number of Null values


# In[334]:


dataset.dropna(inplace=True)


# In[335]:


#Verification of Dropped Column


# In[336]:


dataset.isna().sum()


# In[337]:


#We have succesfully dropped null values 


# In[338]:


#Correlation


# In[339]:


dataset.corr()


# In[340]:


plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(dataset.corr())


# In[341]:


plt.title('Correlation of features', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[342]:


dataset.skew()


# In[343]:


plt.figure(figsize=(14,12), dpi= 80)
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()


# In[344]:


for i in dataset.columns:
    print(i,' -> ',dataset[i].nunique())


# In[345]:


#Dropping name column


# In[346]:


dataset.drop('name',axis=1,inplace=True)


# In[347]:


dataset.head()


# In[348]:


e = dataset.drop('status',axis=1)
f = dataset['status']


# In[349]:


#Splitting the dataset into train and test


# In[350]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(e,f,test_size=0.2,random_state=5,shuffle=True)
print("Splitted the dataset..")


# In[351]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[352]:


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[353]:


ms = []
ms.append(('lr',LogisticRegression()))
ms.append(('lda',LinearDiscriminantAnalysis()))
ms.append(('knn',KNeighborsClassifier()))
ms.append(('dt',DecisionTreeClassifier()))
ms.append(('nb',GaussianNB()))
ms.append(('svm',SVC()))
for name,model in ms:
    kfold = KFold(n_splits=10, shuffle=True, random_state=None)
    results = cross_val_score(model,x_train,y_train,scoring='accuracy',cv=kfold)
    print(name,' -> ', np.mean(results))


# In[ ]:





# In[354]:


from sklearn.preprocessing import StandardScaler
s = StandardScaler()
scaler_xtrain = s.fit_transform(x_train)
scaler_xtrain


# In[355]:


for a, b in mdls:
    kfold = KFold(n_splits=10, shuffle=True)
    results_1 = cross_val_score(b, scaler_xtrain, y_train, cv=kfold, scoring='accuracy')
    print(a, '->', np.mean(results_1))


# In[356]:


from sklearn.model_selection import GridSearchCV
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=10, shuffle=True, random_state=4)


# In[357]:


grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy')


# In[358]:


grid.fit(scaler_xtrain,y_train)


# In[359]:


grid.best_score_


# In[360]:


grid.best_params_


# In[361]:


for index,i in enumerate(grid.cv_results_['mean_test_score']):
    print(grid.cv_results_['params'][index], "->",i)


# In[362]:



param_grid = [{'max_features':[8,9,10,11],'min_samples_split':[2,3,4,5]}]
grid = GridSearchCV(DecisionTreeClassifier(),param_grid)


# In[363]:


grid.fit(scaler_xtrain,y_train)


# In[364]:


grid.best_score_


# In[365]:


grid.best_params_


# In[366]:


from sklearn.ensemble import BaggingClassifier
kfold = KFold(n_splits=10, shuffle=True, random_state=9)
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,random_state=8)
results=cross_val_score(bagging,scaler_xtrain,y_train,cv=kfold,scoring='accuracy')
print(np.mean(results),np.std(results))


# In[367]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
results = cross_val_score(rfc,scal_xtrain,y_train,scoring='accuracy')


# In[368]:


print(np.mean(results),np.std(results))


# In[369]:


from sklearn.ensemble import AdaBoostClassifier
boosting = AdaBoostClassifier(n_estimators=100,random_state=10)
result=cross_val_score(boosting,scaler_xtrain,y_train,cv=kfold,scoring='accuracy')


# In[370]:


print(np.mean(results),np.std(results))


# In[371]:


from sklearn.ensemble import VotingClassifier
model = []
model.append(('knn',KNeighborsClassifier(n_neighbors=1)))
model.append(('svc',SVC()))
model.append(('dt',DecisionTreeClassifier(max_features=9,min_samples_split=4)))
ensemble = VotingClassifier(models)
results = cross_val_score(ensemble,scaler_xtrain,y_train,cv=kfold,scoring='accuracy')


# In[372]:


print(np.mean(results),np.std(results))


# In[373]:


model


# In[374]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
ensemble.fit(scaler_xtrain,y_train)
ypred = ensemble.predict(x_test)


# In[375]:


accuracy_score(y_test,ypred)


# In[376]:


boost=AdaBoostClassifier(n_estimators=100,random_state=16)
boost.fit(scaler_xtrain,y_train)
y_pred_boost = boost.predict(x_test)


# In[377]:


accuracy_score(y_test,y_pred_boost)


# In[378]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(scaler_xtrain,y_train)
y_pred_knn = knn.predict(x_test)


# In[379]:


accuracy_score(y_test,y_pred_knn)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




