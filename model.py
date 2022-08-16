#!/usr/bin/env python
# coding: utf-8

# ## DTSC 691 
# ## Capstone Project
# ## David Bello

# One of the steps in the application as Lead Church Planter is the Church Planter Assessment process. Here, the candidate and spouse go through a series of surveys called the pre-assessments. These are followed by an in-person retreat where the couple is evaluated on areas such as marital satisfaction, expectations, communication skills, etc. After the retreat, the SN assessment team decides if the planter is ready to be endorsed, if he needs further development, or if he is not a good fit for the lead church planter role. Currently, 74% of those invited to the assessment retreat receive the green light to continue in the process.

# This project examines the current pre-assessment process by looking at data from the last three years. It applies statistical analysis and ML methods, particularly of classification, to the existing pre-assessment data to evaluate the effectiveness of the current process, highlight the most significant features in the dataset, and develop an algorithm that learns from the input provided to classify new observations.
#  
# The goal of this project is to develop a classification model that can be used to measure the efficacy of the pre-assessment surveys in determining who should be endorsed as Lead Church Planter.  

# In[1]:


import pandas as pd
import numpy as np


# Import the dataset

# In[2]:


data = pd.read_excel(r"/Users/dbello/Desktop/DTSC_691_Capstone_Project/AP_PlanterAverages.xlsx") #-- work
#data = pd.read_excel(r"/Users/davidbello/Desktop/DTSC691_Capstone_Project/Project_Files/AP_PlanterAverages.xlsx") # home


# In[3]:


data = data.drop(columns=['Idp_Id'])
data


# In[4]:


dataset_4 = data[["Candidate_Observer_Multiplication_Mobilizer",
                  "Candidate_Observer_Multiplication_Discipler",
                  "Candidate_Observer_Multiplication_Server",
                  "Approved"]]


# # ML Model Deployment
# #### Next, we will create a fourth data set of the three most significant features and the target variable and will train a Random Forest Classifier on it. This is the classifier that will be used in the user interface demonstration

# In[5]:


# dataset_4


# In[6]:


# Fill NaN values with mean

dataset_4 = dataset_4.fillna(dataset_4.mean())


# In[7]:


# Split dataset_4 for training and testing
from sklearn.model_selection import train_test_split

X = dataset_4.iloc[:,:-1]
X = X.values # pull the values of the feaures without column names
y = dataset_4['Approved']
y = np.ravel(y, order = 'C')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# In[9]:

from sklearn import svm

# In[10]:

SVM_classifier = svm.SVC()
SVM_classifier.fit(X_train, y_train)

# In[11]:

print(SVM_classifier.score(X_train, y_train)) #prints the R^2 value - how well it fits

prediction_test = SVM_classifier.predict(X_test)
# print(y_test, prediction_test)
print("Mean sq. error between y_test and predicted =", np.mean(prediction_test - y_test)**2)

# In[12]:

print(SVM_classifier.predict([[19, 9, 6]]))

# ## Save Model Using Pickle
# In[13]:


import pickle

# Saving model to disk
pickle.dump(SVM_classifier, open('model.pkl','wb'))




# In[14]:

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[12, 9, 6]])) 

#!pip install flask


# In[15]:


#print(model.predict([[80, 100, 100]])) 


# In[ ]:




