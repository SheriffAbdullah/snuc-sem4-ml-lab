#!/usr/bin/env python
# coding: utf-8

# Use the teleco-customer-churn dataset for the following:
# 
# 1. Apply all the classification algorithms (KNN, Logisitc Regression, Naive Bayes, Decision Trees, SVM) on this dataset and print the accuracies.
# 
# 
# 2. Find out the different tunable parameters for each algorithms mentioned above.
# 
# 
# 3. Apply gridsearchCV and randomizedsearchCV for all the above classification algorithms and get the best parameters.

# <h2>PreProcessing</h2>

# In[1]:


import pandas as pd

data = pd.read_csv('Telco-Customer-Churn.csv')


# In[2]:


data.head()


# In[3]:


data = data.drop('customerID', axis=1)


# In[4]:


# data.info()


# In[5]:


data = data[data['TotalCharges'] != ' ']


# In[6]:


# Convert TotalCharges to 'float64' datatype
data['TotalCharges'] = data['TotalCharges'].astype('float64')


# In[7]:


data.info()
print()


# In[8]:


data.to_csv('cleaned_teleco_customer_churn.csv')
normal_scores = {}
tuned_scores = {}


# In[9]:


# 1-hot encode object Datatypes
obj_columns = data.select_dtypes(exclude=['int64', 'float64']).columns
codes = pd.get_dummies(data[obj_columns], drop_first=True, dtype='int64')
encoded_data = data.join(codes)
encoded_data = encoded_data.drop(obj_columns, axis=1)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

encoded_x = encoded_data.iloc[:, :-1].values
encoded_y = encoded_data.iloc[:, -1].values


# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

scaled_x = sc.fit_transform(encoded_x)


# <h2>Classification Algorithms</h2>
# <h3>1. kNN</h3>

# In[11]:


from sklearn.model_selection import train_test_split
knn_x_train, knn_x_test, knn_y_train, knn_y_test = train_test_split(scaled_x, encoded_y, test_size=0.2, random_state=0)


# In[12]:


#knn_data = data.select_dtypes(include=['int64', 'float64'])

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(knn_x_train, knn_y_train)

knn_y_pred = knn_model.predict(knn_x_test)


# In[13]:


# Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("*** KNN ***")
normal_scores['knn'] = accuracy_score(knn_y_test, knn_y_pred)
print("Accuracy:", normal_scores['knn'])
print()


# In[14]:

print("Confusion Matrix:")
print(confusion_matrix(knn_y_test, knn_y_pred))
print()

# TP FP
# FN TN


# In[15]:


print(classification_report(knn_y_test, knn_y_pred))
print()


# <h3>2. Logistic Regression</h3>

# In[16]:


from sklearn.model_selection import train_test_split
logreg_x_train, logreg_x_test, logreg_y_train, logreg_y_test = train_test_split(scaled_x, encoded_y, test_size=0.25, random_state=0)


# In[17]:


from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()

logreg_model.fit(logreg_x_train, logreg_y_train)


# In[18]:


logreg_y_pred = logreg_model.predict(logreg_x_test)


# In[19]:

print("*** Logistic Regression ***")
normal_scores['logreg'] = accuracy_score(logreg_y_test, logreg_y_pred)
print("Accuracy:", normal_scores['logreg'])
print()


# In[20]:

print("Confusion Matrix:")
print(confusion_matrix(logreg_y_test, logreg_y_pred))
print()


# In[21]:


print(classification_report(logreg_y_test, logreg_y_pred))
print()

# <h3>3. Naive Bayes</h3>

# Source: https://towardsdatascience.com/learning-by-implementing-gaussian-naive-bayes-3f0e3d2c01b2
# 
# If your features are 0 and 1 only, you could use a Bernoulli distribution. 
# 
# If they are integers, a Multinomial distribution. 
# 
# However, we have real feature values and decide for a Gaussian distribution, hence the name Gaussian naive Bayes.
# 
# Also, for Continuous NB Classification refer to: https://remykarem.github.io/blog/naive-bayes

# In[22]:


nb_x_train, nb_x_test, nb_y_train, nb_y_test = train_test_split(scaled_x, encoded_y, test_size=0.25, random_state=0)


# In[23]:


from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(nb_x_train, nb_y_train)


# In[24]:


nb_y_pred = nb_model.predict(nb_x_test)


# In[25]:

print("*** Naive Bayes ***")
normal_scores['nb'] = accuracy_score(nb_y_test, nb_y_pred)
print("Accuracy:", normal_scores['logreg'])
print()


# In[26]:


print("Confusion Matrix:")
print(confusion_matrix(nb_y_test, nb_y_pred))
print()


# In[27]:


print(classification_report(nb_y_test, nb_y_pred))
print()


# <h3>4. Decision Trees</h3>

# In[28]:


dtree_x_train, dtree_x_test, dtree_y_train, dtree_y_test = train_test_split(scaled_x, encoded_y, test_size=0.25, random_state=0)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(criterion='entropy') #or criterion = 'gini'
dtree_model.fit(dtree_x_train, dtree_y_train)


# In[30]:


dtree_y_pred = dtree_model.predict(dtree_x_test)


# In[31]:

print("*** Decision Tree ***")
normal_scores['dtree'] = accuracy_score(dtree_y_test, dtree_y_pred)
print("Accuracy:", normal_scores['dtree'])
print()


# In[32]:

print("Confusion Matrix:")
print(confusion_matrix(dtree_y_test, dtree_y_pred))


# In[33]:


print(classification_report(dtree_y_test, dtree_y_pred))
print()


# <h3>5. SVM</h3>

# In[34]:


from sklearn.svm import SVC

svc_x_train, svc_x_test, svc_y_train, svc_y_test = train_test_split(scaled_x, encoded_y, test_size=0.25, random_state=0)


# In[35]:


svc_model = SVC()

svc_model.fit(svc_x_train, svc_y_train)


# In[36]:


svc_y_pred = svc_model.predict(svc_x_test)


# In[37]:

print("*** SVM ***")
normal_scores['svm'] = accuracy_score(svc_y_test, svc_y_pred)
print("Accuracy:", normal_scores['svm'])
print()


# In[38]:


print("Confusion Matrix:")
print(confusion_matrix(svc_y_test, svc_y_pred))
print()


# In[39]:


print(classification_report(svc_y_test, svc_y_pred))
print()


# <h2> Tunable Parameters </h2>
# 

# In[40]:


import timeit
import warnings
warnings.filterwarnings('ignore')


# <h3>1. kNN</h3>

# In[41]:

# Run below line in Jupyter Notebook to get information about the Function, and its parameters.
# ?KNeighborsClassifier


# In[42]:


# ALL PARAMETERS

knn_parameters = {
    'n_neighbors': [1, 5, 10, 15, 20, 25, 30, 50, 100],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50, 60, 70, 100],
    'p': [1, 2, 3],
    'metric' : ['cityblock', 'euclidean','l1', 'l2', 'manhattan'],
    'n_jobs': [-1]
}

# Metric - 'cosine', 'nan_euclidean', & 'haversine' throw errors.


# In[43]:


# TUNED PARAMETER SET

knn_parameters = {
    'n_neighbors': [25, 50, 75],
    'weights': ['uniform'],
    'algorithm': ['auto', 'kd_tree'],
    'leaf_size': [15, 20, 25],
    'p': [2],
    'metric' : ['eucleidean', 'manhattan', 'l1'],
    'n_jobs': [-1]
}

# Metric - 'cosine', 'nan_euclidean', & 'haversine' throw errors.


# In[44]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
knn_grid = GridSearchCV(knn_model, knn_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)
knn_randomized = RandomizedSearchCV(knn_model, knn_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)


# In[45]:


# In Jupyter Notebook, use the below command to calculate time taken for training each model
# %%timeit -n 1 -r 1
# Source: https://docs.python.org/3/library/timeit.html#timeit-command-line-interface

# 'n' = how many times to execute ‘statement’
# 'r' = how many times to repeat the timer, and calculate average (default 5)

knn_grid.fit(scaled_x, encoded_y)

print()
print(knn_grid.best_params_)
tuned_scores['knn'] = knn_grid.best_score_
print("KNN Accuracy Score: ", tuned_scores['knn'])
print()


# In[46]:


# %%timeit -n 1 -r 1

knn_randomized.fit(scaled_x, encoded_y)

print()
print(knn_randomized.best_params_)
print("KNN Accuracy Score: ", knn_randomized.best_score_)
print()


# <h3>2. Logistic Regression</h3>

# In[47]:

# ?LogisticRegression


# In[48]:


# ALL PARAMETERS

logreg_parameters = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C' : [1e-5, 1e-3, 1e-1, 1, 10, 100, 800, 900, 1000, 1100, 1500],
    'fit_intercept' : [True, False],
    'intercept_scaling': [5, 10, 15],
    'solver': ['auto', 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 600, 700, 800, 900, 1000, 1100, 1200],
    'multi_class': ['auto', 'ovr', 'multinomial'], 
    'n_jobs': [-1]
}


# In[49]:


# TUNED PARAMETER SET

logreg_parameters = {
    'penalty': ['l1', 'none', 'elasticnet'],
    'C' : [1e-175, 100, 800, 1000, 1200],
    'fit_intercept' : [True],
    'intercept_scaling': [True],
    'solver': ['lbfgs', 'liblinear', 'newton-cg'],
    'max_iter': [1000, 1050, 1100, 1500, 2000],
    'multi_class': ['auto', 'multinomial'], 
    'n_jobs': [-1]
}


# In[50]:


logreg_randomized = RandomizedSearchCV(logreg_model, logreg_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)
logreg_grid = GridSearchCV(logreg_model, logreg_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)


# In[51]:


# %%timeit -n 1 -r 1

logreg_randomized.fit(scaled_x, encoded_y)

print()
print(logreg_randomized.best_params_)
print("LogReg Accuracy Score: ", logreg_randomized.best_score_)
print()


# In[52]:


# %%timeit -n 1 -r 1

logreg_grid.fit(scaled_x, encoded_y)

print()
print(logreg_grid.best_params_)
tuned_scores['logreg'] = logreg_grid.best_score_
print("LogReg Accuracy Score: ", tuned_scores['logreg'])
print()


# In[53]:


from sklearn.metrics import get_scorer_names
get_scorer_names()


# In[54]:


# ?GridSearchCV


# <h3>3. Naive Bayes</h3>

# In[55]:


# ?GaussianNB


# In[56]:


nb_parameters = {'var_smoothing': [5, 6, 7, 8, 9, 10]}


# In[57]:


nb_model = GaussianNB()


# In[58]:


nb_randomized = RandomizedSearchCV(nb_model, nb_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)
nb_grid = GridSearchCV(nb_model, nb_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)


# In[59]:


# %%timeit -n 1 -r 1

nb_randomized.fit(scaled_x, encoded_y)

print()
print(nb_randomized.best_params_)
print("Naive Bayes Accuracy Score: ", nb_randomized.best_score_)
print()


# In[60]:


# %%timeit -n 1 -r 1

nb_grid.fit(scaled_x, encoded_y)

print()
print(nb_grid.best_params_)
tuned_scores['nb'] = nb_grid.best_score_
print("Naive Bayes Accuracy Score: ", tuned_scores['nb'])
print()


# <h3>4. Decision Trees</h3>

# In[61]:


# ?DecisionTreeClassifier


# In[62]:


# ALL PARAMETERS

dtree_parameters = {
    'criterion': ["gini", "entropy", "log_loss"],
    'splitter': ["best", "random"],
    'max_depth': [0, 10, 20, 50, 10, 1000, 100000],
    'max_features': [15, 10, 25, 50, 100, 1000, "sqrt", "log2"] # 'auto' is deprecated
}


# In[63]:


# TUNED PARAMETER SET

dtree_parameters = {
    'criterion': ["gini", "entropy", "log_loss"],
    'splitter': ["best", "random"],
    'max_depth': [7, 8, 9],
    'max_features': [11, 12, 13, ] # 'auto' is deprecated
}


# In[64]:


dtree_model = DecisionTreeClassifier() #or criterion = 'gini'


# In[65]:


dtree_randomized = RandomizedSearchCV(dtree_model, dtree_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)
dtree_grid = GridSearchCV(dtree_model, dtree_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)


# In[66]:


# %%timeit -n 1 -r 1

dtree_randomized.fit(scaled_x, encoded_y)

print()
print(dtree_randomized.best_params_)
print("DTree Accuracy Score: ", dtree_randomized.best_score_)
print()


# In[67]:


# %%timeit -n 1 -r 1

dtree_grid.fit(scaled_x, encoded_y)

print()
print(dtree_grid.best_params_)
tuned_scores['dtree'] = dtree_grid.best_score_
print("DTree Accuracy Score: ", tuned_scores['dtree'])
print()


# <h3>5. SVM</h3>

# In[68]:


# ?SVC


# In[69]:


# ALL PARAMETERS

svm_parameters = {
    'kernel': ['linear', 'rbf', 'sigmoid', 'precomputed'],
    'gamma': ['scale', 'auto'],
    'break_ties': [True, False]
}


# In[70]:


# TUNED PARAMETER SET

svm_parameters = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'break_ties': [True, False]
}


# In[71]:


'''
Instead of needlessly computing a lot of SVC fits with different 
degree parameter values, where that will be ignored (all the 
kernels but poly). I suggest splitting the runs for poly and the 
other kernels, you will save a lot of time.

Source:
https://stackoverflow.com/questions/72101295/python-gridsearchcv-taking-too-long-to-finish-running
'''

svm_parameters = {
    'kernel': ['poly'],
    'degree': [1, 2, 3],
    'gamma': ['scale', 'auto'],
    'break_ties': [True, False]
}


# In[72]:


svm_model = SVC()


# In[73]:


svm_randomized = RandomizedSearchCV(svm_model, svm_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)
svm_grid = GridSearchCV(svm_model, svm_parameters, scoring='accuracy', cv=7, n_jobs=-1, verbose=1)


# In[74]:


# %%timeit -n 1 -r 1

svm_randomized.fit(scaled_x, encoded_y)

print()
print(svm_randomized.best_params_)
print("SVM Accuracy Score: ", svm_randomized.best_score_)
print()


# In[75]:


# %%timeit -n 1 -r 1

svm_grid.fit(scaled_x, encoded_y)

print()
print(svm_grid.best_params_)
tuned_scores['svm'] = svm_grid.best_score_
print("SVM Accuracy Score: ", tuned_scores['svm'])
print()


# In[ ]:

print("Increase in accuracy after tuning: ")
print("KNN:", tuned_scores['knn'] - normal_scores['knn'])
print("LogReg:", tuned_scores['logreg'] - normal_scores['logreg'])
print("Naive Bayes:", tuned_scores['nb'] - normal_scores['nb'])
print("DTree:", tuned_scores['dtree'] - normal_scores['dtree'])
print("SVM:", tuned_scores['svm'] - normal_scores['svm'])



