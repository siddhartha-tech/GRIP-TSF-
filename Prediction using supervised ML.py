#!/usr/bin/env python
# coding: utf-8

# # Prediction using Machine Learning
#  
# # Task1 @GRIP21
#  
# # Author-Siddhartha Watsa
#  
# # Predict the percentage of student based on no.of hour of study
# 
# Import libraries and data using data link
# 

# In[14]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following scrip
# 

# In[16]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Pe#rcentage Score')  
plt.show()


# #### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# 
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
# 
# 

# In[17]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# 
# 
# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[18]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# 
# 
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.
# 
# 
# 

# In[19]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[20]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# 
# 
# Now that we have trained our algorithm, it's time to make some predictions.
# 
# 
# 

# In[21]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[22]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ## Graph of Comparison

# In[31]:


df.plot(kind='bar', figsize= (5,5))
plt.grid(which= 'major', linewidth= '0.5')
plt.grid(which= 'minor', linewidth= '0.5')
plt.show()


# ## What will be predicted score is a student studies 9.25 hours/day?

# In[29]:


hours = 9.25
test_new = np.array([hours])
test_new= test_new.reshape(-1,1)
own_pred = regressor.predict(test_new)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# 
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[30]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




