#!/usr/bin/env python
# coding: utf-8

# Use the data.csv and load_iris (from sklearn.datasets)
# 
# Use 3 columns in KMeans - cluster and create 3D graph using seaborn

# <h3>Using data.csv</h3>

# In[1]:

import pandas as pd
data = pd.read_csv('data.csv')

import warnings
warnings.filterwarnings("ignore")

# In[2]:

print(data.head())


# In[3]:

dataset = data.iloc[:, [2, 3, 4]].values


# In[4]:

dataset


# In[5]:

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 21):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(dataset)
    
    wcss.append(km.inertia_)


# In[6]:

import matplotlib.pyplot as plt

plt.plot(range(1, 21), wcss)
plt.title("Elbow-Plateau Graph")
plt.xlabel("Number of means (k)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.show()


# In[7]:

import seaborn as sb
from mpl_toolkits import mplot3d

plot = plt.figure(figsize=(10, 10))
ax = plt.axes(projection ='3d')

x = dataset[:, 0]
y = dataset[:, 1]
z = dataset[:, 2]

ax.scatter(x, z, y, c=x+z)
plt.title("Sample Data")
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.show()


# In[8]:

km = KMeans(n_clusters=6, random_state=0)
y = km.fit_predict(dataset)

y


# In[9]:

colours = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange']

plot = plt.figure(figsize=(10, 10))
ax = plt.axes(projection ='3d')

ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 2], km.cluster_centers_[:, 1], s=150, c='black')
# Changing 'x', 'y', and 'z' arguments to get a different view of the 3D Plot

for i in range(6):
    ax.scatter(dataset[y==i, 0], dataset[y==i, 2], dataset[y==i, 1], c=colours[i])

plt.title("Sample Data")
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.show()

# <h3> Using IRIS Dataset </h3>

# In[10]:

from sklearn.datasets import load_iris
data = load_iris()


# In[11]:

print(data['DESCR'][:1000])
print()


# In[12]:

dataset = data['data']
target = data['target']


# In[13]:

# Columns with highest correllation with Target Class
iris_data = dataset[:, [0, 2, 3]]


# In[14]:

wcss = []

for i in range(1, 21):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(iris_data)
    
    wcss.append(km.inertia_)


# In[15]:

# Elbow Graph
plt.plot(range(1, 21), wcss)

plt.title("IRIS Elbow-Plateau Graph")
plt.xlabel("Number of means (k)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.show()


# In[16]:

# Pretty clearly, in the 'DESCR' of data, there are 3 types of flower
km = KMeans(n_clusters=3, random_state=0)
y = km.fit_predict(iris_data)

y


# In[17]:

colours = ['violet', 'indigo', 'blue']

ax = plt.axes(projection ='3d')
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2], s=150, c='black')

for i in range(3):
    ax.scatter(iris_data[y==i, 0], iris_data[y==i, 1], iris_data[y==i, 2], c=colours[i])

plt.title("IRIS Data")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Petal Length")
ax.set_ylabel("Petal Width")
plt.show()

# In[18]:

target


# In[19]:

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy Score:")
print(accuracy_score(y, target))
print()


# In[20]:

print("Confusion Matrix:")
print(confusion_matrix(y, target))
print()

# ACTUAL CLASS = ROWS
# PREDICTED CLASS = COLUMNS


# In[21]:

print("Classification Report:")
print(classification_report(y, target))
print()


# In[ ]:




