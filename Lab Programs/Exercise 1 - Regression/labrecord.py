# LINEAR REGRESSION

import pandas as pd
import math
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('data1.csv')
print(data)

x = data['x']
y = data['y']

print(data.info())

def simple_linear_regression(x, y):
    n = len(data)

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

    return b0, b1

b0, b1 = simple_linear_regression(x, y)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()

# Regression Line
yPred = b0 + b1 * x

plt.plot(x, yPred)
plt.scatter(x, y, c='r')
plt.show()

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


# MULTIPLE LINEAR REGRESSION

import pandas as pd
data = pd.read_csv('house_pred.csv')

print(data.info())

# Drop columns with majority null values
data = data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Drop ID
data = data.drop(['Id'], axis=1)

data = data.dropna()

obj_columns = data.select_dtypes("object").columns

# Check distribution before null values inputation.
# To decide between mean(), median(), & mode().
import matplotlib.pyplot as plt
data['LotFrontage'].hist()
#data['MasVnrArea'].hist()
#data['GarageYrBlt'].hist()

# Fill-in null values in numeric columns
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].median())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())

print(data.info())

correll = data.corr()
print(correll.style.background_gradient())

# Remove Columns with more than 60% correllation
# Do NOT compare ANY column with Sale Price Column

high_correllation_pairs = []

for i in range(len(correll) - 1):
    for j in range(len(correll) - 1):
        temp = {correll.columns[i], correll.columns[j]}

        if abs(correll.iloc[i,j]) > 0.6 and i != j and temp not in high_correllation_pairs:
            high_correllation_pairs.append(temp)

print(high_correllation_pairs)

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
(high_correllation_columns)

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


x = data.iloc[:,:-1]
y = data.iloc[:, -1]

# Categorical columns = 'object' type columns 
# Make it -> 1-hot encoded

codes = pd.get_dummies(x[obj_columns], drop_first=True, dtype='int64')
x = x.join(codes)
x = x.drop(obj_columns, axis=1)
# Since there are same categories in few columns,
# 'Value = <col_name> + Value' to make categories unique
# Will be done by get_dummies() automatically

x['intercept'] = 1

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['variable']=x.columns

vif['vif'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

max_value = 0
max_column = ''
min_vif_threshold = 20

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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                random_state=0)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

print(reg.intercept_)
print(reg.coef_)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_test,y_pred)

import math

n = len(x)
p = len(x.columns) - 1
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("n =", n)
print("p =", p)
print("r2 =", r2)
print("adj_r2 =", adj_r2)
print("RMSE =", root_mean_squared_error(pd.Series(y_test), pd.Series(y_pred)))
