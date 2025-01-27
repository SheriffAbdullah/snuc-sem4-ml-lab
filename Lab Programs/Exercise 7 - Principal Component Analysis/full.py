# <h4> Download MNIST dataset, apply PCA from scratch. </h4>
# 
# Dataset Source: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
# 
# <h4><u>References:</u></h4>
# 
# https://builtin.com/data-science/step-step-explanation-principal-component-analysis
# 
# https://www.askpython.com/python/examples/principal-component-analysis
# 
# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
# 
# https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
# 
# For plotting:
# https://www.statology.org/scree-plot-python/
# 
# <br>
# 
# Note: TRY BLACKBOX IMPLEMENTATION
# 
# Link: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

data_labelled = pd.read_csv('MNIST/mnist_train.csv')


# In[3]:

print(data_labelled.head())
print()


# In[4]:

label = data_labelled['label']
data = data_labelled.drop('label', axis=1)


# <h5> Data Scaling not required, since all features are of same 0-255 'pixel' values </h5>

# In[5]:

# Make it a numpy 2D Array
data = np.array(data)


# In[6]:

print(f"Range of Pixel values: ({data.min()}, {data.max()})\n")


# In[7]:

print(f"Number of Images: {data.shape[0]} \nNumber of Pixels per Image: {data.shape[1]}\n")


# In[8]:

def GetImage(data, img_no):
    '''
    *** Parameters ***
        data : MNIST CSV Dataset from source mentioned at top
        img_no : Index of the image in data
        
    *** Returns ***
        label : Label of the image
        matrix : Matrix with 28x28 pixel values
    '''
    
    matrix = []
    pixel = 1

    for i in range(0, 28):
        tmp = []
        row_len = 0

        while row_len < 28:
            tmp.append(data.iloc[img_no, pixel])
            pixel += 1
            row_len += 1

        matrix.append(tmp)
        
    return (data.iloc[img_no, 0], matrix)


# In[9]:

img = GetImage(data_labelled, img_no=0)

plt.imshow(img[1], aspect=1)
plt.title("Label:" + str(img[0]))
plt.show()


# In[10]:

# Center columns by subtracting by Mean
# In this case, no significant difference by centering the data.

# data_centered = data - np.mean(data, axis=0)


# In[11]:

# If rowvar is True (default), then each row represents a variable, with observations in the columns
covariance = np.cov(data, rowvar=False)

covariance.shape


# In[25]:

# np.eigh v/s np.eig: https://stackoverflow.com/questions/45434989/numpy-difference-between-linalg-eig-and-linalg-eigh
eigen_values, eigen_vectors = np.linalg.eigh(covariance)


# In[26]:

# 'np.eigh' returns eigenvalues & eigenvectors in ascending sorted order
eigen_values[::-1]


# In[27]:

eigen_values = eigen_values[::-1]

# Reverse the ordering of vectors, to their corresponding eigen values.
eigen_vectors = eigen_vectors[::-1]


# In[28]:

eigen_vectors.shape


# In[29]:

variance_explained = []

for i in eigen_values:
    variance_explained.append(i * 100 / sum(eigen_values))


# In[30]:

# for i in variance_explained:
    # print(i)


# In[63]:

# SCREE PLOT
x = [i for i in range(len(variance_explained))]
plt.plot(x, variance_explained, 'o-', linewidth=1, markersize=2.1)

plt.title("Scree Plot")
plt.xlabel("PCA")
plt.ylabel("% Variance Explained")

plt.show()


# In[38]:

target = 90

cusum = []

s = 0
for i in range(len(variance_explained)): 
    if s > target:
        cols = variance_explained[:i]
        break
        
    s += variance_explained[i]
    cusum.append(s)

print(f"{len(cols)} columns explain {int(s)}% of the variance in the data\n")


# In[39]:

eigen_vectors = eigen_vectors[:len(cols)]


# <h4> Transform the Data using Eigen Vectors </h4>

# In[40]:

print(f"{data.shape} x {eigen_vectors.T.shape}\n")


# In[61]:

data_transformed = np.dot(data, eigen_vectors.T)

print("Transformed Dataset's shape:", data_transformed.shape)


# In[ ]:

# Save the Standard Scaler Object, and the Eigen Vector
# SO that you may 'scale' the input vector, and 'transform' it using Eigen Vector, and get prediction for it.

