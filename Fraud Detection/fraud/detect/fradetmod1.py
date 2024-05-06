#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score,roc_curve,confusion_matrix


# In[124]:


instagram_df_train=pd.read_csv("C:\\Users\\sreek\\Desktop\\django projects\\fraud detection\\Fake-Instagram-Profile-Detection\\insta_train.csv")


# In[58]:


# print(instagram_df_train)


# In[125]:


# instagram_df_test=pd.read_csv("C:\\Users\\sreek\\Desktop\\django projects\\fraud detection\\Fake-Instagram-Profile-Detection\\insta_test.csv")


# In[61]:


# instagram_df_test


# In[8]:


instagram_df_train.isnull().sum()


# In[126]:


x_train=instagram_df_train.drop(columns=['fake'])
# x_test=instagram_df_test.drop(columns=['fake'])


# In[112]:


# print(x_test)


# In[127]:


y_train=np.array(instagram_df_train['fake'])
# y_test=np.array(instagram_df_test['fake'])
# print(y_test.shape)


# In[128]:


#scaling the data before training the model
from sklearn.preprocessing import StandardScaler
scaler_x=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
# x_test=scaler_x.fit_transform(x_test)
# print(x_test)


# In[129]:


# one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)
# print(y_test.shape)


# In[73]:


# print(y_train.shape)


# In[130]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))

model.summary()


# In[131]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[132]:


epochs_hist = model.fit(x_train, y_train, epochs = 50,  verbose = 1, validation_split = 0.1)


# In[133]:


# print(x_test)


# In[78]:


# predicted = model.predict(x_test)


# In[81]:


# predicted_value = []
# test = []
# for i in predicted:
#     predicted_value.append(np.argmax(i))
    
# for i in y_test:
#     test.append(np.argmax(i))
# a,b=0,0
# for i in range(len(y_test)):
#     if predicted_value[i]==test[i]:
#         a+=1
#     b+=1
# # print((a/b)*100)


# In[50]:





# In[86]:


# print(model.predict(x_test))


# In[118]:


# from sklearn.preprocessing import StandardScaler
# scaler_x=StandardScaler()

# d={'profile pic':1,'nums/length username':1,'fullname words':1,'nums/length fullname':1,'name==username':1,'description length':4,'external URL':1,'private':1,'#posts':20,'#followers':555,'#follows':500}
# x_test=x_test.append(d,ignore_index=True)
# x_test=scaler_x.fit_transform(x_test)
# x_test


# In[134]:


# print(model.predict(m_test))
# print(x_test)


# In[143]:


# # print(np.argmax(model.predict(m_test)))

# var=x_test
# var


# In[152]:


def predict(l):
    instagram_df_test=pd.read_csv("C:\\Users\\sreek\\Desktop\\django projects\\fraud detection\\Fake-Instagram-Profile-Detection\\insta_test.csv")
    var=instagram_df_test.drop(columns=['fake'])
    d={'profile pic':l[0],'nums/length username':l[1],'fullname words':l[2],'nums/length fullname':l[3],'name==username':l[4],'description length':l[5],'external URL':l[6],'private':l[7],'#posts':l[8],'#followers':l[9],'#follows':l[10]}
    var=pd.concat([var,pd.DataFrame(d,index=[0])],ignore_index=True)
    var=scaler_x.fit_transform(var)
    predicted = model.predict(var)
    return np.argmax(predicted[len(var)-1])


# In[155]:


# res=(predict([1,0.33,2,0.1,0,5,0,1,8,185,190]))
# if res==1:
#     context={'success':True}
# else:
#     context={'success':False}
# print(context)


# In[ ]:




