#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:41:12 2017

@author: duxuewei
"""
#Must run "supervised_learning.py" and "ranking_supervised.py" before this script

import os
os.chdir('/Users/duxuewei/Desktop/projectds/project/GetOldTweets-python-master/')
import got
import datetime
import pandas as pd



import pandas as pd
import numpy as np
from sklearn import feature_extraction
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import numpy
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import decomposition
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
from sklearn.externals import joblib
import sys
import nltk
from nltk.corpus import stopwords
import re
import os

import math


#Read in pickel files about how we cleaned and transformed the data, and the model
path = '/Users/duxuewei/Desktop/projectds/project/supervised_learning/'
vectorizer = joblib.load(path + 'vectorizer.pk') 
pca = joblib.load(path + 'pca.pk') 
for file in os.listdir(path):
    if file.endswith(".pkl"):
        theFile = file
model = joblib.load(path + theFile) #manual for now    

def genCorpus(theText):
    #set dictionaries
    stopWords = set(stopwords.words('english'))
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
    
    #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words
    tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces       

    return tokens

theCols = ["actors_play", "soundtrack", "filmmaker", "movie_script"]
theLabels = theCols


def class_pred(tempText):
    testText = list()
    testText.append(genCorpus(tempText))
    test = vectorizer.transform(testText)
    X2_new = pca.transform(test.toarray())
    x = model.predict(X2_new)
    x_proba = max(model.predict_proba(X2_new).tolist()[0])
    return (theLabels[x[0]], x_proba)

os.chdir('/Users/duxuewei/Desktop/projectds/project/')


#Read in the data after ranking
tweets_data = pd.read_csv("output.csv")

#Do prediction using our model
prediction_list = list()
prediction_proba = list()
drop_list = list()
for i in tweets_data.index:
    company = tweets_data.loc[i]["text"]
    if isinstance(company, float) == False:
        prediction = class_pred(company)
        prediction_list.append(prediction[0])
        prediction_proba.append(prediction[1])
    else:
        drop_list.append(i)

tweets_data = tweets_data.drop(drop_list)
tweets_data["prediction"] = pd.Series(prediction_list)
tweets_data["prediction_proba"] = pd.Series(prediction_proba)

###We keep rows with probability >0.25 ONLY
tweets_data = tweets_data[tweets_data.prediction_proba > 0.25]


#Split the entire dataset into different categories for better visualization
filtered = tweets_data
sent = filtered.sent.tolist()
text_type = filtered.prediction.tolist()

type1_pos = pd.DataFrame()
type2_pos = pd.DataFrame()
type3_pos = pd.DataFrame()
type0_pos = pd.DataFrame()
type4_pos = pd.DataFrame()

type1_neg = pd.DataFrame()
type2_neg = pd.DataFrame()
type3_neg = pd.DataFrame()
type0_neg = pd.DataFrame()
type4_neg = pd.DataFrame()



for i in range(0,len(filtered)):
    if text_type[i] == theLabels[0]:
        if sent[i] == 'Pos':
            type0_pos = pd.concat([type0_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type0_neg = pd.concat([type0_neg,filtered[i:i+1]])

    elif text_type[i] == theLabels[1]:
        if sent[i] == 'Pos':
            type1_pos = pd.concat([type1_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type1_neg = pd.concat([type1_neg,filtered[i:i+1]])
            
    elif text_type[i] == theLabels[2]:
        if sent[i] == 'Pos':
            type2_pos = pd.concat([type2_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type2_neg = pd.concat([type2_neg,filtered[i:i+1]])
            
    elif text_type[i] == theLabels[3]:
        if sent[i] == 'Pos':
            type3_pos = pd.concat([type3_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type3_neg = pd.concat([type3_neg,filtered[i:i+1]])
    else:
        if sent[i] == 'Pos':
            type4_pos = pd.concat([type4_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type4_neg = pd.concat([type4_neg,filtered[i:i+1]])
            
