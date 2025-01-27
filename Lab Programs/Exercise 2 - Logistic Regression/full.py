# In[ ]:

import pandas as pd
data = pd.read_csv('telecom_customer_churn.csv')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:

print(data.head())
print()


# In[ ]:

data.info()
print()


# In[ ]:

# Drop unwanted columns (or) features
data = data.drop(['Customer ID', 'Churn Category', 'Churn Reason', 'Latitude', 'Longitude', 'Zip Code'], axis=1)


# In[ ]:

obj_columns = data.select_dtypes('object').columns


# In[ ]:

# Check distribution before null values inputation.
# To decide between mean(), median(), & mode().

import matplotlib.pyplot as plt
data['Avg Monthly Long Distance Charges'].hist()
plt.xlabel("Avg Monthly Long Distance Charges")
plt.ylabel("Frequency")
plt.show()

data['Avg Monthly GB Download'].hist()
plt.xlabel("Avg Monthly GB Download")
plt.ylabel("Frequency")
plt.show()
plt.show()


# In[ ]:

# Fill-in null values in numeric columns
data['Avg Monthly Long Distance Charges'] = data['Avg Monthly Long Distance Charges'].fillna(data['Avg Monthly Long Distance Charges'].mean())
data['Avg Monthly GB Download'] = data['Avg Monthly GB Download'].fillna(data['Avg Monthly GB Download'].median())


# In[ ]:

data = data.dropna()


# In[ ]:

# Dichotomous data = Only 2 Categories. E.g. Yes/No


# In[ ]:

# Features
x = data.iloc[:, :-1]

# Target Variables
y = data.iloc[:, -1]


# In[ ]:

# Label Encoding the 'object' type columns
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
for col in obj_columns:
    x[col] = le.fit_transform(x[col])


# In[ ]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x = sc.fit_transform(x)


# In[ ]:

correll = data.corr()
correll.style.background_gradient()


# In[ ]:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=56)


# In[ ]:

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

reg.fit(x_train, y_train)


# In[ ]:

x_train


# In[ ]:

y_pred = reg.predict(x_test)


# In[ ]:

from sklearn.metrics import confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

# TP  FP
# FN  TN


# In[ ]:

from sklearn.metrics import classification_report, accuracy_score

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
print()


# In[ ]:

print("Classification Report:")
print(classification_report(y_test, y_pred))

