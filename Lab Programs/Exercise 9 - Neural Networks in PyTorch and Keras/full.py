#!/usr/bin/env python
# coding: utf-8

# Download the dataset from the following link:
# 
# https://www.kaggle.com/datasets/krantiswalke/bank-personal-loan-modelling
# 
# 
# 
# Create a NN using PyTorch and Keras and compare the final results. You may also use the vanilla NN that you built in AI course and compare the final results.

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[3]:

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")


# In[4]:
    
print(data.head())
print()


# In[5]:

data.info()
print()


# In[6]:

data.corr()


# In[7]:

data = data.drop(['ID', 'ZIP Code'], axis=1)


# In[8]:

x = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]]
y = data.iloc[:, 7]


# In[9]:

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


# <h3>PyTorch</h3>

# In[10]:

import torch
import torchvision
from torchvision import transforms

print("*** Training a Neural Network in PyTorch ***")
print()


# In[11]:

train = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))

test = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))


# In[12]:

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


# In[13]:

# Create the Neural Network in PyTorch

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(11, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return F.log_softmax(self.fc4(x), dim=1)

net = Net()
print(net)
print()


# In[14]:

# Check the Neural Net's 'forward()' Method

x = torch.rand([1, 11])
output = net.forward(x)
output


# In[15]:
    
# print(x)


# In[16]:

def CheckAccuracy():
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testset:
            X, y = data
            output = net(X)
            
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    
    return round(correct/total, 3)


# In[17]:

# Training the Neural Network

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 5
accu_per_epoch = []
loss_per_epoch = []

total = 0
correct = 0

for epoch in range(EPOCHS):
    for data in trainset:
        net.zero_grad()
        
        X, y = data
        output = net(X)
        
        loss = F.nll_loss(output, y)
        loss.backward()
        
        optimizer.step()
    
    accu = CheckAccuracy()
    
    # Verbose
    
    print(f"Epoch: {epoch + 1} | Loss = {loss}, Accuracy = {accu}")
    # For Graphs
    accu_per_epoch.append(accu)
    loss_per_epoch.append(loss)
            


# In[18]:
    
# Plots

loss_per_epoch = [float(x) for x in loss_per_epoch]
plt.plot(range(len(accu_per_epoch)), accu_per_epoch, label="Accuracy")
plt.plot(range(len(loss_per_epoch)), loss_per_epoch, label="Loss")

plt.title("PyTorch: Loss & Accuracy per Epoch")
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[19]:

# Model Accuracy Evaluation

py_test_acc = CheckAccuracy()
print(f'Test accuracy: {py_test_acc:.3f}')
print()


# <h3>TensorFlow</h3>

# In[20]:

import tensorflow as tf

print("*** Training a Neural Network in TensorFlow ***")
print()

# Neural Network Creation
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(30, activation='relu', input_shape=(11,)),
  tf.keras.layers.Dense(30, activation='relu'),
  tf.keras.layers.Dense(30, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[21]:

# Model Training
history = model.fit(x_train, y_train, epochs=5, batch_size=10, verbose=1)


# In[22]:

# Model Accuracy Evaluation

tf_test_loss, tf_test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {tf_test_loss:.3f}, Test accuracy: {tf_test_acc:.3f}')
print()

# In[23]:

# Graphs - Loss & Accuracy for each Epoch

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')

plt.title("TensorFlow: Loss & Accuracy per Epoch")
plt.xlabel('Epoch')
plt.legend()
plt.show()


# <h4>*** After 5 Epochs ***</h4>

# In[24]:

print("*** PyTorch v/s TensorFlow Comparison ***")
print("PyTorch Accuracy:\t", py_test_acc)
print("TensorFlow Accuracy:\t", round(tf_test_acc, 3))


# <h3><u>PyTorch v/s TensorFlow Comparison:</u></h3>
# 
# <p>PyTorch is FASTER in execution time, and
#     
# PyTorch gives almost EQUAL Testing Accuracy as TensorFlow. Note, this is a small dataset.</p>

# <h3>ANN from Scratch</h3>

# In[25]:

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    p = softmax(y_pred)
    
    log_likelihood = -np.log(p[range(m),y_true])
    loss = np.sum(log_likelihood) / m
    
    return loss

def accuracy(y_true, y_pred):
    
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = y_true
    
    return np.sum(pred_labels == true_labels) / true_labels.shape[0]


# In[26]:

print("\n*** Training a Neural Network from Scratch ***")
print()
    
input_dim = x_train.shape[1]
hidden_dim = 30
output_dim = 2
learning_rate = 0.001
epochs = 500


class NeuralNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))

        self.W3 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b3 = np.zeros((1, hidden_dim))

        self.W4 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b4 = np.zeros((1, output_dim))

        self.learning_rate = learning_rate

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = relu(self.z3)

        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.output = softmax(self.z4)

        return self.output

    def backward(self, x, y_true, y_pred):
        m = y_true.shape[0]

        d_output = y_pred
        d_output[range(m), y_true] -= 1
        d_output /= m

        d_z4 = d_output

        d_a3 = np.dot(d_z4, self.W4.T)
        d_z3 = d_a3 * (self.a3 > 0)

        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * (self.a2 > 0)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (self.a1 > 0)

        d_W4 = np.dot(self.a3.T, d_z4)
        d_b4 = np.sum(d_z4, axis=0, keepdims=True)

        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W4 -= self.learning_rate * d_W4
        self.b4 -= self.learning_rate * d_b4

        self.W3 -= self.learning_rate * d_W3
        self.b3 -= self.learning_rate * d_b3

        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1

    def train(self, x_train, y_train, epochs):
        
        losses = []
        accuracies = []
        
        for i in range(epochs+1):
            y_pred = self.forward(x_train)
            loss = cross_entropy_loss(y_train, y_pred)
            acc = accuracy(y_train, y_pred)
            
            losses.append(loss)
            accuracies.append(acc)
            
            self.backward(x_train, y_train, y_pred)
            if i % 100 == 0:
                print("Epoch %d - Loss: %.5f, Accuracy: %.5f" % (i, loss, acc))
        return losses, accuracies


# In[27]:

nn = NeuralNetwork(input_dim, hidden_dim, output_dim, learning_rate)
losses, accuracies = nn.train(x_train, y_train, epochs)


# In[28]:
    
plt.plot(accuracies, label="Accuracy")
plt.plot(losses, label="Loss")

plt.title('ANN from Scratch: Loss & Accuracy per Epoch')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[30]:

y_pred_test = nn.forward(x_test)
ann_test_acc = accuracy(y_test, y_pred_test)
print("Test Accuracy: %.5f" % ann_test_acc)
print()


# <h3>PyTorch & TensorFlow v/s ANN from Scratch Comparison</h3>
# 
# <p>
# It takes many more epochs to get a decent accuracy. It is still lower than PyTorch & TensorFlow. Currently, the neural network is using gradient descent as an optimizer. Using a different optimizer such as Adam may help the neural network converge faster.
# </p>

# <h4> Accuracy after 5 Epochs is very low. </h4>

# <h4> *** After 500 Epochs *** </h4>

# In[31]:

print("ANN Accuracy:\t", ann_test_acc)


# In[32]:

import matplotlib.pyplot as plt

accuracies = [py_test_acc, tf_test_acc, ann_test_acc]

plt.plot(["PyTorch", "TensorFlow", "ANN"], accuracies)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()


# In[ ]:




