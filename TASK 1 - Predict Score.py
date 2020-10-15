#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data imported...")
s_data.head(10)


# In[3]:


s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage')
plt.show()


# In[4]:


X = s_data.iloc[:, :-1].values
Y = s_data.iloc[:, 1].values


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print("Training done...")


# In[7]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, Y)
plt.plot(X, line)
plt.show()


# In[8]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df


# In[14]:


hours = 9.25
own_pred = regressor.predict(np.array([hours]).reshape(1, 1))
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[15]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))


# In[ ]:




