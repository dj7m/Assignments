#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


pip install scikit-learn


# In[6]:


wine=pd.read_csv('C:\\Users\\user\\3D Objects\\winenew.csv')


# In[7]:


wine.info()


# In[8]:


wine


# In[9]:


wine['quality'].value_counts()


# In[10]:


X=wine.iloc[:,:-1]
X.head()


# In[11]:


y=wine.iloc[:,-1]
y.head()


# In[12]:


from sklearn import preprocessing
X=preprocessing.StandardScaler().fit_transform(X)
X[0:1]


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)


# In[16]:


model.fit(X_train,y_train)


# In[32]:


y_prediction=model.predict(X_test)
y_prediction


# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[19]:


acc=accuracy_score(y_test,y_prediction)
acc


# In[20]:


print(model.score(X_test,y_test))


# In[30]:


winee.info()


# In[35]:


y_prediction.size


# In[36]:


from  sklearn.metrics import accuracy_score
from  sklearn.metrics import confusion_matrix


# In[38]:


acc=accuracy_score(y_test,y_prediction)
acc


# In[39]:


cm=confusion_matrix(y_test,y_prediction)
cm


# In[40]:


cm1=pd.DataFrame(data=cm,index=['good','bad'],columns=['good','bad'])
cm1


# In[42]:


prediction_output =pd.DataFrame(data=[y_test.values,y_prediction],index=['y_test','y_prediction'])


# In[43]:


prediction_output.transpose()


# In[23]:


ks =21
mean_acc = np.zeros((ks-1))

for n in range(1,ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat = neigh.predict(X_test) 
    mean_acc[n-1]=accuracy_score(y_test,yhat)
    


# In[44]:


print(mean_acc)


# In[25]:


print("The best accuracy was with",mean_acc.max(),"with k=",mean_acc.argmax()+1)


# In[26]:


plt.plot(range(1,ks),mean_acc,'g')
plt.ylabel('Accuracy')
plt.xlabel('K')
plt.tight_layout()
plt.show()


# In[ ]:




