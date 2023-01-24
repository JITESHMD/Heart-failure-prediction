#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df=pd.read_csv("C:/Users/mdjit/Downloads/heart.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.head(10)


# In[6]:


df["FastingBS"].isnull().sum()


# In[7]:


df["FastingBS"].unique()


# In[8]:


df["Age"].isnull().sum()


# In[9]:


df["Sex"].isnull().sum()


# In[10]:


df["ChestPainType"].isnull().sum()


# In[11]:


df["RestingBP"].isnull().sum()


# In[12]:


df["Cholesterol"].isnull().sum()


# In[13]:


df["RestingECG"].isnull().sum()


# In[14]:


df["MaxHR"].isnull().sum()


# In[15]:


df["ExerciseAngina"].isnull().sum()


# In[16]:


df["Oldpeak"].isnull().sum()


# In[17]:


df["ST_Slope"].isnull().sum()


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])


# In[19]:


df


# In[20]:


df["ChestPainType"].unique()


# In[21]:


le=LabelEncoder()
df['ChestPainType']=le.fit_transform(df['ChestPainType'])


# In[22]:


df


# In[23]:


df["RestingECG"].unique()


# In[24]:


le=LabelEncoder()
df['RestingECG']=le.fit_transform(df['RestingECG'])


# In[25]:


df


# In[26]:


le=LabelEncoder()
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])


# In[27]:


df


# In[28]:


df["ST_Slope"].unique()


# In[29]:


le=LabelEncoder()
df['ST_Slope']=le.fit_transform(df['ST_Slope'])


# In[30]:


df


# In[31]:


x=df.iloc[:,:11]


# In[32]:


x


# In[33]:


y=df['HeartDisease']


# In[34]:


y


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.925)


# In[36]:


x_test.head(10)


# In[37]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()


# In[38]:


model.fit(x_train,y_train)


# In[39]:


x_test[:10]


# In[40]:


y_test[:10]


# In[41]:


model.predict(x_test[:10])


# In[42]:


model.score(x_test,y_test)


# In[43]:


input_data=(65,0,1,149,341,1,2,125,0,2.5,2)

input_arr=np.array(input_data)


# In[48]:


inputs_array=input_arr.reshape(1,-1)
inputs_array


# In[50]:


if (model.predict(inputs_array))==1:
    print("YOUR HEART HAS GOT FAILURE")
else:
    print("YOUR HEART IS NOT FAILURED")


# In[51]:


import pickle


# In[52]:


filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))


# In[54]:


loaded_model=pickle.load(open("trained_model.sav",'rb'))


# In[55]:


if (loaded_model.predict(inputs_array))==1:
    print("YOUR HEART HAS GOT FAILURE")
else:
    print("YOUR HEART IS NOT FAILURED")


# In[ ]:




