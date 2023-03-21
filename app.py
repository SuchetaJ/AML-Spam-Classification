from flask import Flask, jsonify, request, redirect, url_for
from flask import Flask
from score import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import pytest
from sklearn import svm
import sklearn

import requests

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

#splitting the datframe into X and y
X_train = train_df['Message']
y_train = train_df['Label']
X_val = val_df['Message']
y_val = val_df['Label']
X_test = test_df['Message']
y_test = test_df['Label']

tfidf = TfidfVectorizer()

train_tfidf = tfidf.fit_transform(X_train)

loaded_model = pickle.load(open('best_model.sav', 'rb'))


def output(text):
    prediction,propensity = score(text,loaded_model,0.5)
    return prediction,propensity

app = Flask(__name__)

@app.route("/")
def home():
    return "SVM Prediction"

@app.route('/', methods=["POST"])
def predict():
    if request.method == 'POST':
        input_json = request.get_json(force=True) 
        return redirect(url_for('score', text = input_json))
    
@app.route('/score/<text>')
def score(text):  
    prediction,propensity = output(text)
    #propensity = score(X_test[1],loaded_model,0.5)[1]
    dictToReturn = {'prediction':prediction, 'propensity':propensity}
    return jsonify(dictToReturn)


if __name__ == '__main__':
    app.run(port=5000, debug=True, host ='0.0.0.0')