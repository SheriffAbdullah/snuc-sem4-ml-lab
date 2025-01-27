# Use the classification.csv file and compute the Gini index for age and salary column.

# In[2]:

import pandas as pd

data = pd.read_csv("classification.csv")


# In[5]:

print(data.head())
print()


# In[3]:

# Gini Impurity Index: 1 - (p1^2 + p2^2 + ...)
# p = Probability of a unique element occurring.

# Measure of Diversity. More Diversity, More value of Gini Index.
# In simple terms, it is the probability of
# 2 randomly picked (even the same) elements being distinct.

def GiniIndex(col_name, df):
    N = len(df)

    unique_values = df[col_name].unique()
    col = list(df[col_name])

    tmp = 0
    for val in unique_values:
        p = col.count(val) / N
        tmp +=  p ** 2

    return 1 - tmp


# In[4]:

def WeightedGiniIndex(val, col_name, target_column_name, df):
    N = len(df)
    
    left_df = df[df[col_name] < val]
    right_df = df[df[col_name] > val]
    
    p_left = len(left_df) / N
    p_right = len(right_df) / N
    
    return p_left * GiniIndex(target_column_name, left_df) + p_right * GiniIndex(target_column_name, right_df)


# In[5]:

def MinimumWeightedGiniIndex(col_name, target_column_name, df):
    N = len(df)
    
    unique_values = df[col_name].unique()
    unique_values.sort()

    unique_values = list(unique_values)
    
    min_gini = 1
    for i in range(len(unique_values)-1):
        avg = (unique_values[i+1] + unique_values[i]) / 2
        
        gini = WeightedGiniIndex(avg, col_name, target_column_name, df)
        
        if gini < min_gini:
            min_gini = gini
            min_split_value = avg
    
    return min_split_value, min_gini


# In[6]:

print(MinimumWeightedGiniIndex('Age', 'Purchased', data))
print()


# In[7]:

print(f"Gini index of age: {MinimumWeightedGiniIndex('Age', 'Purchased', data)[1]}")
print(f"Gini index of salary: {MinimumWeightedGiniIndex('EstimatedSalary', 'Purchased', data)[1]}")
print()

# 2. Create decision tree algorithm from scratch without using 'sklearn' library. You may assume that all the columns in the data will be categorical in nature. Give a new data for prediction and print the predicted output along with the probabilities.

# In[8]:

data = pd.read_csv('golf_df.csv')


# In[11]:

print(data.head())
print()


# In[10]:

data['Windy'] = data['Windy'].replace(False, 'Weak')
data['Windy'] = data['Windy'].replace(True, 'Strong')


# In[12]:

# Shannon Enropy 'H(S)', measure of uncertainty or randomness in data.
# Entropy = 0 -> No Randomness, accurately predictable.
# Smaller entropy, smaller uncertainity & vice-versa.

import math

def ShannonEntropy(col_name, df):
    N = len(df)

    unique_values = df[col_name].unique()
    col = list(df[col_name])

    tmp = 0
    for val in unique_values:
        p = col.count(val) / N
        tmp += p * math.log2(1 / p)

    return tmp


# In[13]:

# Information Gain: Effective change in entropy after
# deciding on a particular attribute.
# Resource: https://victorzhou.com/blog/information-gain/

def InformationGain(priori_col_name, col_name, df):
    N = len(df)

    unique_values = df[col_name].unique()
    col = list(df[col_name])

    tmp = 0
    for val in unique_values:
        p = col.count(val) / N
        tmp += p * ShannonEntropy(priori_col_name, df[df[col_name] == val])
    
    return ShannonEntropy(priori_col_name, df) - tmp


# In[14]:

print("Shannon Entropy of Data:")
print(ShannonEntropy('Play', data))
print()


# In[15]:

print("Information Gain of Data with respect to 'Windy':")
print(InformationGain('Play', 'Windy', data))
print()


# In[16]:

tree = {}

def C50Algorithm(col_name, data):
    # If all examples are +ve / -ve
    if len(data[col_name].unique()) == 1:
        return data[col_name].unique()[0]

    else:
        # Calculate attribute which has maximum IG
        max_ig = 0
        max_ig_col = ''

        for c in data.columns:
            if c == col_name:
                continue

            tmp = InformationGain(col_name, c, data)

            if tmp > max_ig:
                max_ig = tmp
                max_ig_col = c

        # Create a node with that column
        t = {}
        t[max_ig_col] = {}
        
        # Create Tree
        for i in data[max_ig_col].unique():
            # Remove datapoints corresponding to max IG column's values
            temp_col = data[data[max_ig_col] == i]
            temp_col = temp_col.drop(max_ig_col, axis=1)
            t[max_ig_col][i] = C50Algorithm(col_name, temp_col)
            
        return t


# <u> Graph Answer Output: </u>
# 
# {'Outlook': {"sunny": , "overcast": , "rainy": }}
#  |
#  v
#  
# {'Outlook': {"sunny": {"Humidity": {"normal":, "high":}}, "overcast": , "rainy": }}
#  |
#  v
#  
# {'Outlook': {"sunny": {"Humidity": {"normal":"Yes", "high":"No"}}, "overcast": , "rainy": }}
#  |
#  v
#  
# {'Outlook': {"sunny": {"Humidity": {"normal":"Yes", "high":"No"}}, "overcast": "Yes", "rainy": }}
#  |
#  v
#  
# {'Outlook': {"sunny": {"Humidity": {"normal":"Yes", "high":"No"}}, "overcast": "Yes", "rainy": {'Windy': {'Weak': 'yes', 'Strong': 'no'}}}}
# 

# In[17]:

tree = C50Algorithm('Play', data)
print("Tree: \n", tree)
print()


# In[18]:

def PredictOutput(val, tree):
    if (type(tree) == str):
        return tree
    else:
        for k,v in tree.items():
            # Traverse the node
            tree = v[val[k]]
            del val[k]
            return PredictOutput(val, tree)


# In[19]:

data_point = {"Outlook": 'sunny', "Temperature": 'hot', "Humidity": 'high', "Windy": 'Weak'}

print("Data: \n", data_point)

print("\nPredicted Output:", PredictOutput(data_point, tree))
print()


# In[19]:

# Visualize Tree

import pydot

def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, k+'_'+v)

graph = pydot.Dot(graph_type='graph')
visit(tree)

# Save as image
graph.write_png('example1_graph.png')


# In[1]:

# Try GraphViz



