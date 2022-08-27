#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 16:12:23 2022

@author: abhinav
"""

# Importing the libraries

from flask import Flask, render_template, request
import numpy as np 
from tensorflow import keras

app = Flask(__name__, template_folder="template")

model  = keras.models.load_model("model.h5")

@app.route("/", methods=["GET"])
def home():
    return render_template("/index.html")


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        
        Pregnancies = int(request.form["pregnancies"])
        Glucose = int(request.form["Glucose"])
        BloodPressure = int(request.form["blood_pressure"])
        SkinThickness = int(request.form["skin_thickness"])
        Insulin = int(request.form["insulin"])
        BMI = int(request.form["bmi"])
        DiabetesPedigreeFunction = int(request.form["dpf"]) 
        Age = int(request.form["age"])
        
        
        features = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ]).reshape(1,-1)
        
        prediction = model.predict(features)
        
        if prediction == 1:
            return render_template("/index.html", prediction_text = "YOU ARE DIABETIC")
        else:
            return render_template("/index.html", prediction_text = "YOU ARE NOT DIABETIC")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
