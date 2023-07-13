#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


# Load the Iris dataset
iris_data = pd.read_csv('Iris (4).csv')


# In[5]:


# Separate features (X) and target variable (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Perform feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


# Train the SVM classifier
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)


# In[9]:


# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




