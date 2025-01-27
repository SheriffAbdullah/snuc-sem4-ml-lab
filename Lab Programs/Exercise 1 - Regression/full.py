
# **Exercise 1:** Use the 'data1.csv' to build a simple linear regression from scratch without using sklearn libraries and print the RMSE and mean absolute error values.
# 

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('data1.csv')


# In[3]:


x = data['x']
y = data['y']

data.info()
print()


# Using 'y = b0 + (b1 * x)', and partial derivation to find values of 'b0' and 'b1' for minimum value of SSE.
# 

# In[4]:


def simple_linear_regression(x, y):
    n = len(x)

    # Sum(x * y)
    xy_sum = 0
    for i in range(n):
        xy_sum += x.iloc[i] * y.iloc[i]

    # Sum(x^2)
    xSq_sum = 0
    for i in range(n):
        xSq_sum += x.iloc[i] * x.iloc[i]

    b1 = (xy_sum - n * x.mean() * y.mean()) / (xSq_sum - n * x.mean() * x.mean())
    b0 = y.mean() - b1 * x.mean()
    
    print("\nUsing Direct Derivative formula to minimize Error:")
    print("Intercept, b0 =", b0)
    print("Slope, b1 =", b1)
    
    
    # Calculating the SAME parameter, 'm' using Karl-Pearson's method
    x_deviation_sq_sum = 0
    y_deviation_sq_sum = 0
    x_deviation_y_deviation_sum = 0
    for i in range(n):
        x_deviation_sq_sum += (x.iloc[i] - x.mean()) ** 2
        y_deviation_sq_sum += (y.iloc[i] - y.mean()) ** 2
        x_deviation_y_deviation_sum += (x.iloc[i] - x.mean()) * (y.iloc[i] - y.mean())
    
    # Standard Deviation of the populaton uses 'n' in the denominator.
    # If you're taking a sample of the population, use 'n-1' in the denominator.
    sx = math.sqrt((1 / n) * x_deviation_sq_sum)
    sy = math.sqrt((1 / n) * y_deviation_sq_sum)
        
    # Karl-Pearson's Correllation Coefficient
    r = x_deviation_y_deviation_sum / (math.sqrt(x_deviation_sq_sum) * math.sqrt(y_deviation_sq_sum))

    # Slope
    b1 = r * sy / sx
    
    print("\nUsing Karl-Pearson's formula:")
    print("Intercept, b0 =", b0)
    print("Slope, b1 =", b1)
    print("Correllation Coefficient, r =", r)
    print("Standard Deviation of x, sx =", sx)
    print("Standard Deviation of y, sy =", sy)
    
    return b0, b1


# In[5]:


b0, b1 = simple_linear_regression(x, y)


# In[6]:


plt.scatter(x, y)
plt.title("Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[7]:


# Regression Line
yPred = b0 + b1 * x


# In[8]:


plt.plot(x, yPred)
plt.scatter(x, y, c='r')
plt.title("Regression Line for Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[9]:


# Error Metrics

def sum_squared_error(y, yPred):
    n = len(y)

    SSE = 0
    for i in range(n):
        SSE += (y.iloc[i] - yPred.iloc[i]) ** 2

    return SSE 

def mean_squared_error(y, yPred):
    n = len(y)

    SSE = sum_squared_error(y, yPred)
    MSE = SSE / n

    return MSE

def root_mean_squared_error(y, yPred):
    MSE = mean_squared_error(y, yPred)
    RMSE = math.sqrt(MSE)   

    return RMSE

def mean_absolute_error(y, yPred):
    n = len(y)

    MAE = 0
    for i in range(n):
        MAE += abs(y.iloc[i] - yPred[i])

    MAE /= n

    return MAE

def mean_percentage_error(y, yPred):
    if len(y) != len(yPred):
        raise Exception(f"length of 'y' & 'yPred' are not matching ({len(y), len(yPred)})")
    
    n = len(y)
    MPE = 0

    for i in range(n):
        MPE += abs(y.iloc[i] - yPred.iloc[i]) / y.iloc[i]

    MPE /= n
    MPE *= 100
    
    return MPE


# In[10]:


# Error
# e = y - yPred

# Sum of Squared Errors
SSE = sum_squared_error(y, yPred)

# Mean Squared Error
MSE = mean_squared_error(y, yPred)

# Root Mean Squared Error
RMSE = root_mean_squared_error(y, yPred)

# Mean Absolute Error
MAE = mean_absolute_error(y, yPred)

# Mean Percentage Error
MPE = mean_percentage_error(y, yPred)

print(f"RMSE: {RMSE}")
print(f"MAE: {MAE}")


# <h4> Using Gradient Descent </h4>

# In[11]:


class LinearRegressionScratch():
    
    def __init__(self, no_of_iters=1000, learning_rate=0.001):
        self.no_of_iters = no_of_iters
        self.learning_rate = learning_rate
        self.cost_history = []
        
    
    def fit(self, x, y):
        self.m, self.n = x.shape
        
        self.x = x
        self.y = y
        
        self.b0 = 0
        self.b1 = np.zeros(self.n,)

        for _ in range(self.no_of_iters):
            self._update_weights()
            self._calculate_cost()
            
        
    def predict(self, x):
        y_pred = self.b0 + np.dot(x, self.b1)
        return y_pred
        
        
    def _update_weights(self):
        self.y_pred = self.predict(self.x)
        
        db0 = -2 * np.mean(self.y - self.y_pred)
        db1 = -2 * np.mean(self.x.T * (self.y - self.y_pred))

        self.b0 = self.b0 - self.learning_rate * db0
        self.b1 = self.b1 - self.learning_rate * db1
        
        
    def _calculate_cost(self):
        # Mean of Squared Error
        y_pred = self.predict(self.x)
        loss = np.mean((y_pred - self.y) ** 2)
        self.cost_history.append(loss)
        
    
    def get_cost(self):
        return self.cost_history
    
    
    def get_params(self):
        return self.b0, self.b1[0]
        
        
# NOTE: Cost v/s Loss
# 'loss' is the loss function, say, Binary Cross Entropy or SSE.
# 'cost' is the summation of 'loss' for all data points.


# In[12]:


x = np.array(x)
y = np.array(y)

# In reshape, '-1' = Number of existing datapoints
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


# In[13]:


model = LinearRegressionScratch(no_of_iters=100, learning_rate=0.003)
model.fit(x, y)

y_pred = model.predict(x)

# NOTE: You may perform Hyperparameter Tuning to find optimim value for 'learning_rate'


# In[16]:


cost = model.get_cost()


# In[17]:


plt.plot(range(len(cost)), cost)

plt.title("Cost v/s Number of Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()


# In[18]:


b0, b1 = model.get_params()

print("\nUsing Gradient Descent:")
print("Intercept, b0 =", b0)
print("Slope, b1 =", b1)


# In[29]:


from sklearn.metrics import mean_squared_error

print("RMSE =", math.sqrt(mean_squared_error(y, y_pred)))

# Gradient Descent gets stuck on local optimas, and is not accurate in this case.
# With larger datasets, it will be more helpful, because it takes less time for Computation.


# In[30]:


plt.figure(figsize=(7, 7))

plt.scatter(x, y)
plt.plot(x, yPred)
plt.plot(x, y_pred, c='r')

plt.title("Gradient Descent v/s Direct Regression Formula")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(['Datapoints', 'Minimized Error using Derivatives', 'Gradient Descent'])
plt.show()


# **Exercise 2:** Use the 'house_pred.csv' file to build a multiple linear regression model. 'sklearn' shall be used to fit the model. Perform necessary preprocessing and check for outliers and multi-collinearity.
# 
# Apply the same set of preprocessing to the 'test.csv' and use the data to predict the house price. The evaluation criteria will be Root Mean Squared Error.

# In[ ]:


import pandas as pd
import warnings

data = pd.read_csv('house_pred.csv')

# To avoid warning messages in output
warnings.filterwarnings('ignore')


# In[ ]:

print()
data.info()


# In[ ]:


# Drop columns with majority null values
data = data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Drop ID
data = data.drop(['Id'], axis=1)


# In[ ]:


obj_columns = data.select_dtypes("object").columns


# In[ ]:


# Remove all null values in 'object' type columns

data = data.dropna(subset=obj_columns)
data
print()


# In[ ]:


# Check distribution before null values inputation.
# To decide between mean(), median(), & mode().
import matplotlib.pyplot as plt
data['LotFrontage'].hist()
#data['MasVnrArea'].hist()
#data['GarageYrBlt'].hist()


# In[ ]:


# Fill-in null values in numeric columns
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].median())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())


# In[ ]:


correll = data.corr()
correll.style.background_gradient()


# In[ ]:


# Remove Columns with more than 60% correllation
# Do NOT compare ANY column with Sale Price Column

high_correllation_pairs = []

for i in range(len(correll) - 1):
    for j in range(len(correll) - 1):
        temp = {correll.columns[i], correll.columns[j]}

        if abs(correll.iloc[i,j]) > 0.6 and i != j and temp not in high_correllation_pairs:
            high_correllation_pairs.append(temp)

print("High Correllation Pairs:")
print(high_correllation_pairs)
print()


# In[ ]:


# Columns with high correllation with other columns
# Elements = (Column Name, Correllation with Sale price)
high_correllation_columns = []

for pair in high_correllation_pairs:
    temp = list(pair)
    for col in temp:
        t = data[col].corr(data.iloc[:, -1])
        out = [col, t]
        if out not in high_correllation_columns:
           high_correllation_columns.append(out) 

# Sort by the correllation value
high_correllation_columns.sort(key=lambda a:a[1])

print("Correllation of Features with Target Variable:")
print(high_correllation_columns)
print()


# In[ ]:


# Remove columns with low correllation with 'y' variable, and have high correllation with other variables.
# TODO: Optimize Code here
hcp = high_correllation_pairs.copy()
tmp = hcp.copy()

while hcp:
    for col in high_correllation_columns:
        c = 0
        for j in tmp:
            if col[0] in j:
                try:
                    hcp.remove(j)
                except:
                    pass
                if c == 0:
                    data = data.drop(col[0], axis=1)
                    c+=1
        tmp = hcp.copy()

print("Removing columns with low correllation with target variable, and high correllation with other features.")
print()

# In[ ]:


# Manual Method
#data = data.drop(columns=['2ndFlrSF', 'GarageYrBlt', 'GarageArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF', 'BsmtFullBath', 'YearRemodAdd'], axis=1)


# In[ ]:


x = data.iloc[:,:-1]
y = data.iloc[:, -1]


# In[ ]:


data.columns
print()


# In[ ]:


# Categorical columns = 'object' type columns 
# Make it -> 1-hot encoded

codes = pd.get_dummies(x[obj_columns], drop_first=True, dtype='int64')
x = x.join(codes)
x = x.drop(obj_columns, axis=1)
# Since there are same categories in few columns,
# 'Value = <col_name> + Value' to make categories unique
# Will be done by get_dummies() automatically


# In[ ]:


'''
for col in obj_columns:
    
    for i in range(len(x)):
        x[col].iloc[i] = col + x[col].iloc[i]

    # Get the 1-hot encoding for every 'object' column
    dummies = pd.get_dummies(x[col], drop_first=True)
    x = x.join(dummies)

    # Drop the old column
    x = x.drop(col, axis=1)
'''


# In[ ]:


x['intercept'] = 1


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['variable']=x.columns

vif['vif'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]


# In[ ]:


max_value = 0
max_column = ''
min_vif_threshold = 20

print(f"Removing features with VIF > {min_vif_threshold} using VIF Removal Algorithm.")

while True:
    for value in vif.values:
        if (value[1] > max_value and value[1] != vif.values[-1][1]):
            max_value = value[1]
            max_column = value[0]

    if (max_value < min_vif_threshold):
        break

    print(max_column, max_value)
    x = x.drop(max_column, axis=1)

    # Calculate VIF
    vif = pd.DataFrame()
    vif['variable']=x.columns
    vif['vif'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

    max_value = 0

print()

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(x_train, y_train)


# In[ ]:


y_pred = reg.predict(x_test)


# In[ ]:


print("Intercept:", reg.intercept_)
print("Slope Values:", reg.coef_[-5:])
print()


# In[ ]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_test,y_pred)


# In[ ]:


import math

n = len(x)
p = len(x.columns) - 1
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print('Error Metrics:')
print("n =", n)
print("p =", p)
print("r2 =", r2)
print("adj_r2 =", adj_r2)
print("RMSE =", root_mean_squared_error(pd.Series(y_test), pd.Series(y_pred)))


# In[ ]:




