#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data1=pd.read_csv('C:\\Users\\asus\\Downloads\\iris (2).csv',encoding = "ISO-8859-1")
data1.head()
data1["species"]=data1["species"].map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
from sklearn.model_selection import train_test_split
x=data1[["sepal_length","sepal_width","petal_length","petal_width"]]
y=data1["species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=30)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("Accuracy=",accuracy_score(y_pred,y_test))


# In[10]:


import pandas as pd
data1=pd.read_csv('C:\\Users\\asus\\Downloads\\iris (2).csv',encoding = "ISO-8859-1")
data1.head()

