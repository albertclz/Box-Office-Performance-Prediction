# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 00:02:40 2017

@author: Jack. Wang
"""

import os
import pandas as pd
import re
import numpy as np
import sys
import string
import nltk.data
import tweepy
import logging
import pickle
import matplotlib
from datetime import datetime
from dateutil import parser
from tweepy import OAuthHandler
from textblob import TextBlob
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer


ranking_whole = pd.read_csv('ranking_whole.csv')


def genCorpus(theText):
    stopWords = set(stopwords.words('english'))
    #theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
        
        #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", token) for token in tokens]
    #tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words
    #tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    
    tokens = ' '.join(tokens) #need to pass string seperated by spaces       
    
    return tokens

sid = SentimentIntensityAnalyzer()
def SID_Analyzer(text):
    result =  sid.polarity_scores(text)
    return result


ranking_whole['Compound_Score']= 'NA'; ranking_whole['Negative_Score']= 'NA'; ranking_whole['Positive_Score']= 'NA'; ranking_whole['Neutral_Score']= 'NA'
cnt=0
for i in ranking_whole['0']:
    try:
        info = genCorpus(i)
        if info != '':
            ranking_whole['0'][cnt] = info
        else:
            ranking_whole['0'][cnt] = 'neutral'
        compound = SID_Analyzer(ranking_whole['0'][cnt])
        ranking_whole['Compound_Score'][cnt] = compound['compound']
        ranking_whole['Negative_Score'][cnt] = compound['neg']
        ranking_whole['Positive_Score'][cnt] = compound['pos']
        ranking_whole['Neutral_Score'][cnt] = compound['neu']
        cnt +=1
    except:
        ranking_whole['0'][cnt] = 'neutral'
        compound = SID_Analyzer(ranking_whole['0'][cnt])
        ranking_whole['Compound_Score'][cnt] = compound['compound']
        ranking_whole['Negative_Score'][cnt] = compound['neg']
        ranking_whole['Positive_Score'][cnt] = compound['pos']
        ranking_whole['Neutral_Score'][cnt] = compound['neu']
        cnt +=1

ranking_whole.to_csv('ranking_whole.csv', index = False, encoding = 'utf-8')
