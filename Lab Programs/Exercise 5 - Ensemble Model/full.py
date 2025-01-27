#!/usr/bin/env python
# coding: utf-8

# Use any classification dataset used previously and create your own ensemble of classification models. Use minimum 5 different classification algos in your code and submit it.

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

data = pd.read_csv('cleaned_teleco_customer_churn.csv')


# In[3]:

data = data.drop('Unnamed: 0', axis=1)


# In[4]:

print(data.head())
print()


# In[5]:

# 1-hot encode object Datatypes
obj_columns = data.select_dtypes(exclude=['int64', 'float64']).columns
codes = pd.get_dummies(data[obj_columns], drop_first=True, dtype='int64')
encoded_data = data.join(codes)
encoded_data = encoded_data.drop(obj_columns, axis=1)

encoded_x = encoded_data.iloc[:, :-1].values
encoded_y = encoded_data.iloc[:, -1].values


# In[6]:

# SCALE THE FEATURE VECTORS

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_x = sc.fit_transform(encoded_x)


# In[7]:

# INSTANTIATE THE MODELS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def InstantiateModels():
    knn_model = KNeighborsClassifier(n_neighbors=5)
    logreg_model = LogisticRegression()
    nb_model = GaussianNB()
    dtree_model = DecisionTreeClassifier(criterion='entropy')
    svm_model = SVC()
    
    return [knn_model, logreg_model, nb_model, dtree_model, svm_model]


# In[8]:

models = InstantiateModels()


# In[9]:

# BAGGING

# Samples with replacement
def bag(x, y, bag_size=0.7):
    # Bag Size
    N = int(data.shape[0] * bag_size)
    
    # np.random.randint(low, high, size), where size = size of matrix with random numbers
    indices = np.random.randint(0, data.shape[0], data.shape[0])

    # Source: https://pandas.pydata.org/docs/user_guide/indexing.html
    out = (pd.DataFrame(x, index=indices), 
           pd.DataFrame(y, index=indices))
    
    for df in out:
        df = df.reset_index(drop=True)
        df = df.iloc[:N]
    
    return out[0], out[1]


# In[10]:

# MODEL TRAINING

def EnsembleTrain(x, y, models):
    for model in models:
        sample_x, sample_y = bag(x, y)
        
        '''
        DataConversionWarning: A column-vector y was passed when 
        a 1d array was expected. Please change the shape of y to 
        (n_samples, ), for example using ravel().
        '''
        sample_y = pd.Series.ravel(sample_y)
        
        model.fit(sample_x, sample_y)
        
    return models


# In[11]:

models = EnsembleTrain(scaled_x, encoded_y, models)


# In[12]:

def MaxVote(predictions):
    y_pred = []

    for value in range(len(predictions[0])):
        temp = 0

        for pred_model in range(len(predictions)):
            temp += predictions[pred_model][value]

        if temp > 2: y_pred.append(1)
        else: y_pred.append(0)
            
    return y_pred


# In[13]:

# PREDICTION

def EnsembleClassifier(x_test, models):
    predictions = []

    # PREDICT OUTPUT FOR EACH MODEL
    for model in models:
        pred = model.predict(x_test)
        predictions.append(pred)
    
    # MAX VOTING
    y_pred = MaxVote(predictions)
    
    return y_pred


# In[14]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_x, encoded_y, test_size=0.2, random_state=0)


# In[15]:

y_pred = EnsembleClassifier(x_test, models)


# In[16]:

from sklearn.metrics import accuracy_score

print(f"The ensemble accuracy is {accuracy_score(y_test, y_pred)}")
print()


# In[17]:

print("*** Individual Accuracies ***")
for model in models:
    score = accuracy_score(y_test, model.predict(x_test))
    print(model, ":", score)
    
