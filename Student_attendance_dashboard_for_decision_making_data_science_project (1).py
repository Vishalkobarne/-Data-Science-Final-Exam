#!/usr/bin/env python
# coding: utf-8

# In[455]:


#Importing nessesary libraries


# In[456]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
print("Libraries installed Successfully......")


# In[457]:


#Importing dataset_1


# In[458]:


dataset_1 = pd.read_csv(r'C:\Users\Kiran\Downloads\Batchwise Attendance Data - Class 1.csv')
print("dataset_1 Loaded Successfully...")


# In[459]:


dataset_1.head()


# In[460]:


dataset_1.tail()


# In[461]:


#Checking shape of the dataset_1


# In[462]:


dataset_1.shape


# In[463]:


#Full Data Summary


# In[464]:


dataset_1.info()


# In[465]:


#Statistical Summary of Data


# In[466]:


dataset_1.describe()


# In[467]:


#Finding out Null values in Each Columns


# In[468]:


dataset_1.isna().sum()


# In[469]:


#Total number of NaN values in the dataset


# In[470]:


dataset_1.isna().sum().sum()


# In[471]:


#Importing dataset_2


# In[472]:


dataset_2 = pd.read_csv(r'C:\Users\Kiran\Downloads\Batchwise Attendance Data - Class 2.csv')
print("dataset_2 Loaded Successfully...")


# In[473]:


dataset_2.head()


# In[474]:


dataset_2.tail()


# In[475]:


#Checking shape of the dataset_2
dataset_2.shape


# In[476]:


#Full Data Summary


# In[477]:


dataset_2.info()


# In[478]:


# Statistical Summary of Data


# In[479]:


dataset_2.describe()


# In[480]:


#Finding out Null values in Each Columns


# In[481]:


dataset_2.isna().sum()


# In[482]:


#Total number of NaN values in the dataset


# In[483]:


dataset_2.isna().sum().sum()


# In[484]:


dataset_2['Type'].replace(['NaN'],'student')


# In[485]:


dataset_2['Type'] = dataset_2['Type'].fillna('STUDENT')


# In[486]:


dataset_1['Type'] = dataset_1['Type'].fillna('STUDENT')


# In[487]:


dataset_1.isna().sum()


# In[488]:


dataset_2.isna().sum()


# In[489]:


dataset_2.drop(['01-01-21','R-01-01-21'],axis=1,inplace=True)


# In[490]:


dataset_2.isna().sum()


# In[491]:


dataset_2.head()


# In[492]:


from sklearn.impute import SimpleImputer


# In[493]:


imputer = SimpleImputer(strategy='most_frequent')
new_data = imputer.fit_transform(dataset_1)


# In[494]:


new_data


# In[495]:


dataset_1 = pd.DataFrame(new_data,columns = dataset_1.columns)


# In[496]:


dataset_1.head()


# In[497]:


imputer = SimpleImputer(strategy='most_frequent')
new_data2 = imputer.fit_transform(dataset_2)


# In[498]:


dataset_2 = pd.DataFrame(new_data2,columns=dataset_2.columns)


# In[499]:


dataset_2.head()


# In[500]:


dataset_1.shape


# In[501]:


dataset_2.shape


# In[502]:


dataset_1['Type'].unique(),data['10-01-21'].unique()


# In[503]:


dataset_1.info()


# In[504]:


dataset_2.info()


# In[505]:


dataset_2['R-10-01-21'].unique()


# In[506]:


# DATAFRAME FOR CLASS A


# In[507]:


date_10 = dataset_1[['Student Roll Num','Type','10-01-21','R-10-01-21']]


# In[508]:


date_9 = dataset_1[['Student Roll Num','Type','09-01-21','R-9-01-21']]


# In[509]:


date_8 = dataset_1[['Student Roll Num','Type','08-01-21','R-8-01-21']]


# In[510]:


date_7 = dataset_1[['Student Roll Num','Type','07-01-21','R-7-01-21']]


# In[511]:


date_6 = dataset_1[['Student Roll Num','Type','06-01-21','R-6-01-21']]


# In[512]:


date_5 = dataset_1[['Student Roll Num','Type','05-01-21','R-5-01-21']]


# In[513]:



date_4 = dataset_1[['Student Roll Num','Type','04-01-21','R-4-01-21']]


# In[514]:


date_3 = dataset_1[['Student Roll Num','Type','03-01-21','R-3-01-21']]


# In[515]:


date_2 = dataset_1[['Student Roll Num','Type','02-01-21','R-02-01-21']]


# In[516]:


date_1 = dataset_1[['Student Roll Num','Type','01-01-21','R-01-01-21']]


# In[517]:


# DATAFRAME FOR CLASS B


# In[518]:


clb_10 = dataset_2[['Student Roll Num','Type','10-01-21','R-10-01-21']]


# In[519]:



clb_9 = dataset_2[['Student Roll Num','Type','09-01-21','R-9-01-21']]


# In[520]:


clb_8 = dataset_2[['Student Roll Num','Type','08-01-21','R-8-01-21']]


# In[521]:


clb_7 = dataset_2[['Student Roll Num','Type','07-01-21','R-7-01-21']]


# In[522]:


clb_6 = dataset_2[['Student Roll Num','Type','06-01-21','R-6-01-21']]


# In[523]:


clb_5 = dataset_2[['Student Roll Num','Type','05-01-21','R-5-01-21']]


# In[524]:


clb_4 = dataset_2[['Student Roll Num','Type','04-01-21','R-4-01-21']]


# In[525]:



clb_3 = dataset_2[['Student Roll Num','Type','03-01-21','R-3-01-21']]


# In[526]:


import scipy.stats


# In[527]:


rat = [5,6,7,8,9,10]
total_rat_cnt =[]
for r in rat:
    cnt =0
    rat_cnt = []
    for i in dataset_1.columns:
        if i[0] == 'R':
            for j in data[i]:
                if j == r:
                    cnt+=1
            rat_cnt.append(cnt)
    total_rat_cnt.append(sum(rat_cnt))
print(total_rat_cnt)


# In[528]:


plt.bar(rat,total_rat_cnt)
plt.title('Total Rating From Different Days')
plt.show()


# In[529]:


Total_Score = []
for i in dataset_1.columns:
    if i[0] == 'R':
        Total_Score.append(scipy.stats.mode(dataset_1[i]))
print(scipy.stats.mode(Total_Score))


# In[530]:


# Rating and Visualisations for Specific Dates for class A
# These are all the students who gave 10 out of 10 rating for class A
plt.pie(date_10[date_10['R-10-01-21']>9]['Type'].value_counts(),labels=date_10[date_10['R-10-01-21']>9]['Type'].unique())
plt.show()
date_10[date_10['R-10-01-21']>9]['Type'].value_counts()


# In[531]:


sns.catplot(x='Type', kind='count', data=dataset_1, col='10-01-21')
plt.show()


# In[532]:


sns.countplot(x='Type',
             hue='R-10-01-21',
             data=dataset_1)


# In[533]:


plt.pie(date_9[date_9['R-9-01-21']>9]['Type'].value_counts(),labels=date_9[date_9['R-9-01-21']>9]['Type'].unique())
plt.show()
date_9[date_9['R-9-01-21']>9]['Type'].value_counts()


# In[534]:


sns.catplot(x='Type', col='09-01-21', kind='count', data=dataset_1)
plt.show()


# In[535]:


sns.countplot(x='Type',
             hue='R-9-01-21',
             data=dataset_1)


# In[536]:


plt.pie(date_8[date_8['R-8-01-21']>9]['Type'].value_counts(),labels=date_8[date_8['R-8-01-21']>9]['Type'].unique())
plt.show()
date_8[date_8['R-8-01-21']>9]['Type'].value_counts()


# In[537]:


sns.catplot(x='Type',col='08-01-21',kind='count',data=dataset_1)
plt.show()


# In[538]:


sns.countplot(x='Type',
             hue='R-8-01-21',
             data=dataset_1)


# In[539]:


plt.pie(date_7[date_7['R-7-01-21']>9]['Type'].value_counts(),labels=date_7[date_7['R-7-01-21']>9]['Type'].unique())
plt.show()
date_7[date_7['R-7-01-21']>9]['Type'].value_counts()


# In[540]:



sns.catplot(x='Type',col='07-01-21',kind='count',data=dataset_1)
plt.show()


# In[541]:


sns.countplot(x='Type',
             hue='R-7-01-21',
             data=dataset_1)


# In[542]:



plt.pie(date_6[date_6['R-6-01-21']>9]['Type'].value_counts(),labels=date_6[date_6['R-6-01-21']>9]['Type'].unique())
plt.show()
date_6[date_6['R-6-01-21']>9]['Type'].value_counts()


# In[543]:


sns.catplot(x='Type',col='06-01-21',kind='count',data=dataset_1)
plt.show()


# In[544]:


sns.countplot(x='Type',
             hue='R-6-01-21',
             data=dataset_1)


# In[545]:


plt.pie(date_5[date_5['R-5-01-21']>9]['Type'].value_counts(),labels=date_5[date_5['R-5-01-21']>9]['Type'].unique())
plt.show()
date_5[date_5['R-5-01-21']>9]['Type'].value_counts()


# In[546]:


sns.catplot(x='Type',col='05-01-21',kind='count',data=dataset_1)
plt.show()


# In[547]:


sns.countplot(x='Type',
             hue='R-5-01-21',
             data=dataset_1)


# In[548]:



plt.pie(date_4[date_4['R-4-01-21']>9]['Type'].value_counts(),labels=date_4[date_4['R-4-01-21']>9]['Type'].unique())
plt.show()
date_4[date_4['R-4-01-21']>9]['Type'].value_counts()


# In[549]:


sns.catplot(x='Type',col='04-01-21',kind='count',data=dataset_1)
plt.show()


# In[550]:


sns.countplot(x='Type',
             hue='R-4-01-21',
             data=dataset_1)


# In[551]:


plt.pie(date_3[date_3['R-3-01-21']>9]['Type'].value_counts(),labels=date_3[date_3['R-3-01-21']>9]['Type'].unique())
plt.show()
date_3[date_3['R-3-01-21']>9]['Type'].value_counts()


# In[552]:


sns.catplot(x='Type',col='03-01-21',kind='count',data=dataset_1)
plt.show()


# In[553]:


sns.countplot(x='Type',
             hue='R-3-01-21',
             data=dataset_1)


# In[554]:


plt.pie(date_2[date_2['R-02-01-21']>9]['Type'].value_counts(),labels=date_2[date_2['R-02-01-21']>9]['Type'].unique())
plt.show()
date_2[date_2['R-02-01-21']>9]['Type'].value_counts()


# In[555]:


sns.catplot(x='Type',col='02-01-21',kind='count',data=dataset_1)
plt.show()


# In[556]:


sns.catplot(x='Type',col='02-01-21',kind='count',data=dataset_1)
plt.show()


# In[557]:



plt.pie(date_1[date_1['R-01-01-21']>9]['Type'].value_counts(),labels=date_1[date_1['R-01-01-21']>9]['Type'].unique())
plt.show()
date_1[date_1['R-01-01-21']>9]['Type'].value_counts()


# In[558]:


sns.catplot(x='Type',col='01-01-21',kind='count',data=dataset_1)
plt.show()


# In[559]:


sns.countplot(x='Type',
             hue='R-01-01-21',
             data=dataset_1)


# In[560]:


Total_Score1 = []
for i in data2.columns:
    if i[0] == 'R':
        Total_Score1.append(scipy.stats.mode(data2[i]))
print(scipy.stats.mode(Total_Score1))


# In[561]:


rat2 = [5,6,7,8,9,10]
total_rat2_cnt =[]
for r in rat2:
    cnt =0
    rat_cnt = []
    for i in data2.columns:
        if i[0] == 'R':
            for j in data2[i]:
                if j == r:
                    cnt+=1
            rat_cnt.append(cnt)
    total_rat2_cnt.append(sum(rat_cnt))
print(total_rat2_cnt)


# In[562]:


plt.bar(rat2,total_rat2_cnt)
plt.title('Total Rating From Different Days')
plt.show()


# In[563]:



plt.pie(clb_10[clb_10['R-10-01-21']>9]['Type'].value_counts(),labels=clb_10[clb_10['R-10-01-21']>9]['Type'].unique())
plt.show()
clb_10[clb_10['R-10-01-21']>9]['Type'].value_counts()


# In[564]:


sns.catplot(x='Type',col='10-01-21',kind='count',data=dataset_2)
plt.show()


# In[565]:


sns.countplot(x='Type',
             hue='R-10-01-21',
             data=dataset_2)


# In[566]:



plt.pie(clb_9[clb_9['R-9-01-21']>9]['Type'].value_counts(),labels=clb_9[clb_9['R-9-01-21']>9]['Type'].unique())
plt.show()
clb_9[clb_9['R-9-01-21']>9]['Type'].value_counts()


# In[567]:


sns.catplot(x='Type',col='09-01-21',kind='count',data=dataset_2)
plt.show()


# In[568]:



sns.countplot(x='Type',
             hue='R-9-01-21',
             data=dataset_2)


# In[569]:



plt.pie(clb_8[clb_8['R-8-01-21']>9]['Type'].value_counts(),labels=clb_8[clb_8['R-8-01-21']>9]['Type'].unique())
plt.show()
clb_8[clb_8['R-8-01-21']>9]['Type'].value_counts()


# In[570]:


sns.catplot(x='Type',col='08-01-21',kind='count',data=dataset_2)
plt.show()


# In[571]:


sns.countplot(x='Type',
             hue='R-8-01-21',
             data=dataset_2)


# In[572]:



plt.pie(clb_7[clb_7['R-7-01-21']>9]['Type'].value_counts(),labels=clb_7[clb_7['R-7-01-21']>9]['Type'].unique())
plt.show()
clb_7[clb_7['R-7-01-21']>9]['Type'].value_counts()


# In[573]:


sns.catplot(x='Type',col='07-01-21',kind='count',data=dataset_2)
plt.show()


# In[574]:



sns.countplot(x='Type',
             hue='R-7-01-21',
             data=dataset_2)


# In[575]:


plt.pie(clb_6[clb_6['R-6-01-21']>9]['Type'].value_counts(),labels=clb_6[clb_6['R-6-01-21']>9]['Type'].unique())
plt.show()
clb_6[clb_6['R-6-01-21']>9]['Type'].value_counts()


# In[576]:


sns.catplot(x='Type',col='06-01-21',kind='count',data=dataset_2)
plt.show()


# In[577]:


sns.countplot(x='Type',
             hue='R-6-01-21',
             data=dataset_2)


# In[578]:


plt.pie(clb_5[clb_5['R-5-01-21']>9]['Type'].value_counts(),labels=clb_5[clb_5['R-5-01-21']>9]['Type'].unique())
plt.show()
clb_5[clb_5['R-5-01-21']>9]['Type'].value_counts()


# In[579]:


sns.catplot(x='Type',col='05-01-21',kind='count',data=dataset_2)
plt.show()


# In[580]:


sns.countplot(x='Type',
             hue='R-5-01-21',
             data=dataset_2)


# In[581]:


plt.pie(clb_4[clb_4['R-4-01-21']>9]['Type'].value_counts(),labels=clb_4[clb_4['R-4-01-21']>9]['Type'].unique())
plt.show()
clb_4[clb_4['R-4-01-21']>9]['Type'].value_counts()


# In[582]:


sns.catplot(x='Type',col='04-01-21',kind='count',data=dataset_2)
plt.show()


# In[583]:


sns.countplot(x='Type',
             hue='R-4-01-21',
             data=dataset_2)


# In[584]:


plt.pie(clb_3[clb_3['R-3-01-21']>9]['Type'].value_counts(),labels=clb_3[clb_3['R-3-01-21']>9]['Type'].unique())
plt.show()
clb_3[clb_3['R-3-01-21']>9]['Type'].value_counts()


# In[585]:


sns.catplot(x='Type',col='03-01-21',kind='count',data=dataset_2)
plt.show()


# In[586]:


sns.countplot(x='Type',
             hue='R-3-01-21',
             data=dataset_2)


# In[ ]:





# In[ ]:




