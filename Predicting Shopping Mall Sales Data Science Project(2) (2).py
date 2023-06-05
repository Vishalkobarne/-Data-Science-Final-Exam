#!/usr/bin/env python
# coding: utf-8

# In[159]:


#Importing nessesary libraries


# In[167]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
print("Libraries installed Successfully......")


# In[161]:


#Importing dataset


# In[168]:


dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/Shopping_Revenue.csv')
print("Dataset Loaded Successfully...")


# In[169]:


dataset.head()


# In[170]:


dataset.tail()


# In[171]:


#Checking shape of the 


# In[172]:


dataset.shape


# In[ ]:


#Full Data Summary


# In[173]:


dataset.info()


# In[ ]:


#Statistical Summary of Data


# In[174]:


dataset.describe()


# In[ ]:


#Checking the datatypes


# In[175]:


dataset.dtypes


# In[ ]:


# Unique columns


# In[176]:


for i in dataset.columns:
    print(i,dataset[i].nunique())


# In[ ]:


#Finding out Null values in Each Columns


# In[177]:


dataset.isna().sum()


# In[ ]:


#Total number of NaN values in the dataset


# In[178]:


dataset.isna().sum().sum()


# In[ ]:


#Dropping of Columns


# In[179]:


dataset.drop(['Id','Open Date','City'],axis=1,inplace=True)


# In[180]:


dataset.dropna(inplace=True)


# In[ ]:


#Verification of Dropped Column


# In[181]:


dataset.isna().sum()


# In[ ]:


# checking unique values of Type column


# In[182]:


dataset['Type'].unique()


# In[183]:


dataset['City Group'] = dataset['City Group'].map({'Big Cities':0,'Other':1})
dataset['Type'] = dataset['Type'].map({'IL':0, 'FC':1, 'DT':2})


# In[184]:


for i in dataset.columns:
    print(np.var(dataset[i]))


# In[216]:


# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(dataset["revenue"], kde=True)
plt.title("Distribution of revenue")
plt.xlabel("revenue")
plt.ylabel("Count")
plt.show()


# In[ ]:


#Correlation 


# In[185]:


dataset.corr()


# In[213]:


plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(dataset.corr(),cmap = 'coolwarm')


# In[189]:


from sklearn.preprocessing import StandardScaler
values = dataset.values
sc = StandardScaler()
y_scale = sc.fit_transform(values[:,39].reshape(-1,1))


# In[190]:


dataset['revenue'] = y_scale


# In[191]:


i = dataset.drop('revenue',axis=1)
j = dataset['revenue']


# In[192]:


x_train,x_test,y_train,y_test = train_test_split(i,j,test_size=0.2,random_state=1)


# In[193]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[194]:


from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
Rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=10)
Decisiontree_model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',Rfe),('m',Decisiontree_model)])
cc = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
Na_scores = cross_val_score(pipeline, x_train, y_train, cv=cc)
Na_scores


# In[195]:


print(np.mean(Na_scores))


# In[196]:


Na_features = [10,13,15,18,22,25,28,32]
for i in Na_features:
    Rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=i)
    Decisiontree_model = DecisionTreeRegressor()
    pipeline = Pipeline(steps=[('s',Rfe),('m',Decisiontree_model)])
    cc = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    Na_scores = cross_val_score(pipeline, x_train, y_train, cv=cc,scoring='r2')
    print(i,np.mean(Na_scores),np.std(Na_scores))


# In[197]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
Rfr = RandomForestRegressor()
Rfe = RFE(estimator=Rfr, n_features_to_select=22)
fs_xtrain = Rfe.fit_transform(x_train, y_train)
print(Rfe.support_)
print(Rfe.ranking_)


# In[198]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# In[199]:


Ms = []
Ms.append(('LR',LinearRegression()))
Ms.append(('knn',KNeighborsRegressor()))
Ms.append(('dt',DecisionTreeRegressor()))
Ms.append(('svm',SVR()))


# In[200]:


for v,k in Ms:
    kfold = KFold(n_splits=3, shuffle=False)
    results_e = cross_val_score(k, x_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    print(v,np.mean(results_e))


# In[201]:


for name,mdl in Ms:
    kfold = KFold(n_splits=3, shuffle=True, random_state=1)
    results_f = cross_val_score(mdl,fs_xtrain,y_train,cv=kfold,scoring = 'neg_mean_squared_error')
    print(name,np.mean(results_f))


# In[202]:


pm = [1,3,5,7,9,11,13,15,17]
for i in pm:
    pipeline_1 = []
    mdl1 = KNeighborsRegressor(n_neighbors=i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    result= cross_val_score(mdl1,x_train,y_train,cv=kfold,scoring = 'neg_mean_squared_error')
    print(i,np.mean(result))


# In[203]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[204]:


ensemble = [('rfc',RandomForestRegressor()),('ab',AdaBoostRegressor()),('gb',GradientBoostingRegressor())]
for x,y in ensemble:
    kfold = KFold(n_splits=3, shuffle=True, random_state=8)
    ensemble_result = cross_val_score(y,fs_xtrain,y_train,cv=kfold,scoring = 'neg_mean_squared_error')
    print(x,np.mean(ensemble_result))


# In[205]:


from sklearn.model_selection import GridSearchCV
sd=7
scoring = 'neg_mean_squared_error'
prm_grd = dict(n_estimators=np.array([50,75,100,125,150,200]))
mdl = GradientBoostingRegressor(random_state=sd)
Kfold = KFold(n_splits=3, shuffle=True, random_state=sd)
grd = GridSearchCV(estimator=mdl, param_grid=prm_grd, scoring=scoring, cv=Kfold)
grd_rslt = grd.fit(x_train, y_train)


# In[206]:


grd.best_score_


# In[207]:


grd.cv_results_['params']


# In[208]:


from sklearn.ensemble import VotingRegressor
st = VotingRegressor([('svm',SVR()),('knn',KNeighborsRegressor(n_neighbors=13))])
k_fold = KFold(n_splits=3, shuffle=True, random_state=98)
rslt_vot = cross_val_score(st,fs_xtrain,y_train,cv=k_fold,scoring='neg_mean_squared_error')


# In[209]:


print(np.mean(rslt_vot))


# In[210]:


from sklearn.metrics import mean_squared_error


# In[211]:


VR = VotingRegressor([('svm',SVR()),('knn',KNeighborsRegressor(n_neighbors=13))])
VR.fit(x_train,y_train)
ypred = VR.predict(x_test)


# In[212]:


mean_squared_error(y_test,ypred)


# In[ ]:





# In[ ]:




