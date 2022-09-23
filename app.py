import json
import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# Creating new app

app=Flask(__name__)           #__name__ is nothing but the starting point of the app from where it will start
regmodel=pickle.load(open('regmodel.pkl','rb'))   #Opening and then loading Pickle file for deployment
scalar=pickle.load(open('scaling.pkl','rb'))

##after localhost address if we put this app.route slash then it will redirect us to home.html page
@app.route('/')     
def home():
    return render_template('home.html')

# Creating predict api to fetch values using Postman or nay other tool
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(data))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

#
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted price of car is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)