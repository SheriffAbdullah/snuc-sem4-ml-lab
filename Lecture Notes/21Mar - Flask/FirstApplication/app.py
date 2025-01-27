from flask import Flask, request, render_template

app = Flask(__name__)

#%%

import pickle

model = pickle.load(open('model.pkl', 'rb'))

#%% 

@app.route("/")
def main():
    return render_template("index.html")

#%% 

@app.route("/predict", methods=["post"])
def pred():
    print(request.form.values()) # Prints a Generator Object 'MultiDict.values'
    
    features = [float(i) 
                for i in 
                request.form.values()]
    # TypeConversion: Since form returns a list of 'str' datatypes
    
    pred = model.predict([features])
    pred = round(pred[0], 2)
    
    return render_template("index.html", data=pred)
    
    #%%
    
if __name__ == '__main__':
    app.run(host='localhost', port=5000) # Default port = 5000

'''
Port Numbers:

TCP = 80 - A Protocol
SMTP = 25 - A Protocol
MySQL= 3306 - An Application
''' 

#%%
# YOUR MODEL

'''
data = pd.read_excel('hiring.xlsx')

x = data.iloc[].values
y = data.iloc[].values

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x, y)

print(reg.predict())
'''

reg = 5

#%%

# ENCODE THE MODEL

import pickle

pickle.dump(reg, open('model.pkl', 'wb'))