# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:39:54 2017

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

os.chdir('D:\Columbia-Course\CS-6998\Project')
os.getcwd()

corpus_13 = pd.read_csv('corpus_13.csv')
corpus_14 = pd.read_csv('corpus_14.csv')
corpus_15 = pd.read_csv('corpus_15.csv')
corpus_16 = pd.read_csv('corpus_16.csv')
corpus_17 = pd.read_csv('corpus_17.csv')
corpus_18 = pd.read_csv('corpus_18.csv')
corpus_19 = pd.read_csv('corpus_19.csv')
corpus_20 = pd.read_csv('corpus_20.csv')
corpus_21 = pd.read_csv('corpus_21.csv')
corpus_22 = pd.read_csv('corpus_22.csv')
corpus_23_24 = pd.read_csv('corpus_23_24.csv')

###SUM of retweet
retweets_13 = sum(corpus_13['retweets'])
np.mean(corpus_13['retweets'])

retweets_14 = sum(corpus_14['retweets'])
retweets_15 = sum(corpus_15['retweets'])
retweets_16 = sum(corpus_16['retweets'])
retweets_17 = sum(corpus_17['retweets'])
retweets_18 = sum(corpus_18['retweets'])
retweets_19 = sum(corpus_19['retweets'])
retweets_20 = sum(corpus_20['retweets'])
retweets_21 = sum(corpus_21['retweets'])
retweets_22 = sum(corpus_22['retweets'])
retweets_23_24 = sum(corpus_23_24['retweets'])

###Mean of favorite
fav_13 = np.mean(corpus_13['favorites'])
fav_14 = np.mean(corpus_14['favorites'])
fav_15 = np.mean(corpus_15['favorites'])
fav_16 = np.mean(corpus_16['favorites'])
fav_17 = np.mean(corpus_17['favorites'])
fav_18 = np.mean(corpus_18['favorites'])
fav_19 = np.mean(corpus_19['favorites'])
fav_20 = np.mean(corpus_20['favorites'])
fav_21 = np.mean(corpus_21['favorites'])
fav_22 = np.mean(corpus_22['favorites'])
fav_23_24 = np.mean(corpus_23_24['favorites'])

#Indicator
corpus_13['Indicator']= 'NA'
corpus_14['Indicator']= 'NA'
corpus_15['Indicator']= 'NA'
corpus_16['Indicator']= 'NA'
corpus_17['Indicator']= 'NA'
corpus_18['Indicator']= 'NA'
corpus_19['Indicator']= 'NA'
corpus_20['Indicator']= 'NA'
corpus_21['Indicator']= 'NA'
corpus_22['Indicator']= 'NA'
corpus_23_24['Indicator']= 'NA'

i=0
corpus_13['Compound_Score']
while i < len(corpus_13['retweets']):
    if corpus_13['Compound_Score'][i] > 0.0516:
        corpus_13['Indicator'][i] = 'PP'
    elif corpus_13['Compound_Score'][i] < -0.0516:
        corpus_13['Indicator'][i] = 'Neg'
    else:
        corpus_13['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_14['Compound_Score']
while i < len(corpus_14['retweets']):
    if corpus_14['Compound_Score'][i] > 0.0516:
        corpus_14['Indicator'][i] = 'PP'
    elif corpus_14['Compound_Score'][i] < -0.0516:
        corpus_14['Indicator'][i] = 'Neg'
    else:
        corpus_14['Indicator'][i] = 'NN'
    i +=1
    
i=0
corpus_15['Compound_Score']
while i < len(corpus_15['retweets']):
    if corpus_15['Compound_Score'][i] > 0.0516:
        corpus_15['Indicator'][i] = 'PP'
    elif corpus_15['Compound_Score'][i] < -0.0516:
        corpus_15['Indicator'][i] = 'Neg'
    else:
        corpus_15['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_16['Compound_Score']
while i < len(corpus_16['retweets']):
    if corpus_16['Compound_Score'][i] > 0.0516:
        corpus_16['Indicator'][i] = 'PP'
    elif corpus_16['Compound_Score'][i] < -0.0516:
        corpus_16['Indicator'][i] = 'Neg'
    else:
        corpus_16['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_17['Compound_Score']
while i < len(corpus_17['retweets']):
    if corpus_17['Compound_Score'][i] > 0.0516:
        corpus_17['Indicator'][i] = 'PP'
    elif corpus_17['Compound_Score'][i] < -0.0516:
        corpus_17['Indicator'][i] = 'Neg'
    else:
        corpus_17['Indicator'][i] = 'NN'
    i +=1
















