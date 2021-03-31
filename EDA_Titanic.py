#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARY

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# # LOAD DATA

# In[2]:


df=pd.read_csv("TitanicDataset.csv")
df.head()


# # DATA VISUALIZATION

# In[3]:


df.isnull()


# In[4]:


sns.countplot(x="Survived", data=df)


# In[5]:


sns.countplot(x="Survived", hue='Sex',data=df)


# In[6]:


sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=60)


# In[7]:


sns.countplot(x="Survived", hue='Pclass',data=df)


# In[8]:


sns.countplot(x="SibSp",data=df)


# In[9]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')


# # INSERT VALUES FOR NAN VALUES

# In[11]:


def input_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 28
        else:
            return 24
    else:
        return Age


# In[12]:


df['Age']=df[['Age','Pclass']].apply(input_age,axis=1)


# In[13]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[14]:


df.drop('Cabin',axis=1,inplace=True)


# In[15]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:


df.head()


# In[17]:


df.isnull()


# # CONVERT CATAGORICAL DATA INTO NUMERICAL DATA

# In[18]:


df.Pclass.value_counts()


# In[19]:


df.Sex.value_counts()


# In[20]:


df.Embarked.value_counts()


# In[21]:


df['Sex']=pd.factorize(df.Sex)[0]


# In[22]:


df


# In[23]:


df.Sex.value_counts()


# In[24]:


df['Embarked']=pd.factorize(df.Embarked)[0]
df


# # FINAL DATA FOR MODELING

# In[25]:


df.drop(['Name','Ticket'],axis=1,inplace=True)
df


# In[34]:


x=df.drop('Survived',axis=1)
x


# In[35]:


y=df['Survived']
y


# # TRAIN TEST SPLIT

# In[36]:


import sklearn 
from sklearn.model_selection import train_test_split


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=101)


# # TRAINING AND PREDICT

# In[38]:


from sklearn.linear_model import LogisticRegression


# In[40]:


mymodel=LogisticRegression()
mymodel.fit(x_train,y_train)


# In[44]:


prediction=mymodel.predict(x_test)


# In[45]:


from sklearn.metrics import confusion_matrix


# In[46]:


accuracy=confusion_matrix(y_test,prediction)
accuracy


# In[52]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,prediction)
accuracy


# In[57]:


df['Survived'].head(20)


# In[58]:


prediction


# In[ ]:




