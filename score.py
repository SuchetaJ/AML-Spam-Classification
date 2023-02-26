from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn import svm
import sklearn 

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

def score(text:str, model: sklearn.svm._classes.SVC, threshold:float) -> tuple:
    propensity = model.predict_proba(tfidf.transform([text]))[0]
    desired_predict = (model.predict_proba(tfidf.transform([text]))[:,1] >= threshold).astype(bool)
    #return (desired_predict[0], propensity)
    return (bool(desired_predict[0]), float(max(propensity)))