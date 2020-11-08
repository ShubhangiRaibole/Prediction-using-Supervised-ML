#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# Reading data from remote link
data = "http://bit.ly/w-data"
df = pd.read_csv(data)

df.head()


# In[9]:


df.shape


# In[10]:


df.describe()


# In[12]:


df.dtypes


# In[17]:


# Plotting the distribution of scores
df.plot(x= "Hours", y = "Scores", style = "*",color= "g")
plt.title ('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[16]:


x=df.iloc[:,:-1]
y=df.iloc[:,1]


# In[31]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.3, random_state=5) 


# In[62]:


x_train


# In[37]:


from sklearn.linear_model import LinearRegression  
simple_linear_regression = LinearRegression()  


# In[38]:


simple_linear_regression.fit(x_train,y_train)


# In[ ]:


# plt.scatter(x_train, y_train, color='red')
plt.plot (x_train,simple_linear_regression.predict(x_train),color='black')
plt.title("hours vs percentage")
plt.xlabel('hourse')
plt.ylabel('Percentage')
plt.show()


# In[43]:


simple_linear_regression.intercept_


# In[44]:


simple_linear_regression.coef_


# # Making Predictions

# In[47]:


y_pred = regressor.predict(x_test)
print(x_test)


# # Comparing Actual vs Predicted

# In[48]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[57]:


# You can also test with your own data
hours=[[9.25]]
pred =float(simple_linear_regression.predict(hours))
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred))


# In[59]:


from sklearn import metrics 


# In[60]:


print('Mean Absolute Error:', 
     metrics.mean_absolute_error(y_test, y_pred))
print('Mean squared Error:', 
     metrics.mean_squared_error(y_test, y_pred)) 
   


# # Thank you
