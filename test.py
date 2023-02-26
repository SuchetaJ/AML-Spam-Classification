from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import pytest
from sklearn import svm
import sklearn
import requests
import app
from app import *

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

from score import *

def test_score():
    threshold =0.5
    label, propensity = score(X_test[10],loaded_model,threshold)
    assert type(X_test[10]) == str
    assert type(loaded_model) == sklearn.svm._classes.SVC
    assert type(threshold) == float
    assert type(label) == bool
    assert type(propensity) == float
    assert (0<=propensity<=1) == True

    #When threshold is set to 0
    threshold = 0
    label, propensity = score(X_test[1],loaded_model,threshold) #testing on non spam
    assert label == True #to check if labels non spam also as spam
    label, propensity = score(X_test[10],loaded_model,threshold) #testing on spam
    assert label == True #to check if correct labels spam

    #When threshold is set to 1
    threshold = 1
    label, propensity = score(X_test[1],loaded_model,threshold)
    assert label == False  #to check if correct labels spam
    label, propensity = score(X_test[10],loaded_model,threshold)
    assert label == False #to check if labels spam also as non spam

    #assertion on Spam SMS
    print("Obvious Spam text")
    for i in range(len(y_test)):
        if y_test[i]==1:
            spam_sms = X_test[i]
            break
    label, propensity = score(spam_sms,loaded_model,0.5)
    assert label == True

    #Assertion on non spam SMS
    print("Obvious Non-Spam text")
    for i in range(len(y_test)):
        if y_test[i]==0:
            nonspam_sms = X_test[i]
            break
    label, propensity = score(nonspam_sms,loaded_model,0.5)
    assert label == False

import time
import os
import signal
import subprocess
def test_flask():
    proc = subprocess.Popen(['python', 'app.py'])
    response = requests.get('http://127.0.0.1:5000/')
    assert response.status_code == 200
    assert response.text == 'SVM Prediction'
    proc.terminate()