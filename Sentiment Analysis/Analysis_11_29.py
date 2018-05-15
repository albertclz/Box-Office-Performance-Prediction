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
from sklearn import metrics, datasets, linear_model
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats
from scipy.stats.stats import *
from inspect import getargspec, ismethod, isclass, formatargspec
import statsmodels.api as sm
from sklearn.neural_network import MLPClassifier

os.chdir('/Users/Egg/Desktop/Data-Science/Project-in-DS/Final_Project/11-29')
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

i=0
corpus_18['Compound_Score']
while i < len(corpus_18['retweets']):
    if corpus_18['Compound_Score'][i] > 0.0516:
        corpus_18['Indicator'][i] = 'PP'
    elif corpus_18['Compound_Score'][i] < -0.0516:
        corpus_18['Indicator'][i] = 'Neg'
    else:
        corpus_18['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_19['Compound_Score']
while i < len(corpus_19['retweets']):
    if corpus_19['Compound_Score'][i] > 0.0516:
        corpus_19['Indicator'][i] = 'PP'
    elif corpus_19['Compound_Score'][i] < -0.0516:
        corpus_19['Indicator'][i] = 'Neg'
    else:
        corpus_19['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_20['Compound_Score']
while i < len(corpus_20['retweets']):
    if corpus_20['Compound_Score'][i] > 0.0516:
        corpus_20['Indicator'][i] = 'PP'
    elif corpus_20['Compound_Score'][i] < -0.0516:
        corpus_20['Indicator'][i] = 'Neg'
    else:
        corpus_20['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_21['Compound_Score']
while i < len(corpus_21['retweets']):
    if corpus_21['Compound_Score'][i] > 0.0516:
        corpus_21['Indicator'][i] = 'PP'
    elif corpus_21['Compound_Score'][i] < -0.0516:
        corpus_21['Indicator'][i] = 'Neg'
    else:
        corpus_21['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_22['Compound_Score']
while i < len(corpus_22['retweets']):
    if corpus_22['Compound_Score'][i] > 0.0516:
        corpus_22['Indicator'][i] = 'PP'
    elif corpus_22['Compound_Score'][i] < -0.0516:
        corpus_22['Indicator'][i] = 'Neg'
    else:
        corpus_22['Indicator'][i] = 'NN'
    i +=1

i=0
corpus_23_24['Compound_Score']
while i < len(corpus_23_24['retweets']):
    if corpus_23_24['Compound_Score'][i] > 0.0516:
        corpus_23_24['Indicator'][i] = 'PP'
    elif corpus_23_24['Compound_Score'][i] < -0.0516:
        corpus_23_24['Indicator'][i] = 'Neg'
    else:
        corpus_23_24['Indicator'][i] = 'NN'
    i +=1
    
###Average Sentiment Score
sen_13 = np.mean(corpus_13['Compound_Score'])
sen_14 = np.mean(corpus_14['Compound_Score'])
sen_15 = np.mean(corpus_15['Compound_Score'])
sen_16 = np.mean(corpus_16['Compound_Score'])
sen_17 = np.mean(corpus_17['Compound_Score'])
sen_18 = np.mean(corpus_18['Compound_Score'])
sen_19 = np.mean(corpus_19['Compound_Score'])
sen_20 = np.mean(corpus_20['Compound_Score'])
sen_21 = np.mean(corpus_21['Compound_Score'])
sen_22 = np.mean(corpus_22['Compound_Score'])
sen_23_24 = np.mean(corpus_23_24['Compound_Score'])

aver_sent = [sen_13, sen_14, sen_15, sen_16, sen_17, sen_18, sen_19, sen_20,
             sen_21, sen_22, sen_23_24]
Aver_sent = pd.DataFrame({'col':aver_sent})

rank = [1,1,1,2,2,2,2,3,3,3,4]
Rank = pd.DataFrame({'col':rank})

Daily_gross = [11659375, 9362760, 5016890, 1366905, 1727445, 1059510, 1116480, 2989520,
               4095415, 2278480, 644240]

fav = [fav_13, fav_14, fav_15, fav_16, fav_17, fav_18, fav_19, fav_20, fav_21, fav_22, 
       fav_23_24 ]
Fav = pd.DataFrame({'col':fav})

retweett = [retweets_13, retweets_14, retweets_15, retweets_16, retweets_17, retweets_18, retweets_19,
            retweets_20, retweets_21, retweets_22, retweets_23_24]
Retweett = pd.DataFrame({'col':retweett})

daily_theatre = [3149, 3149, 3149, 3149, 3149, 3149, 3149, 3298, 3298, 3298, 3298]
Daily_theatre = pd.DataFrame({'col':daily_theatre})


# Create linear regression object
regr = linear_model.LinearRegression()


regr.fit(Aver_sent, Daily_gross)
gross_pred = regr.predict(Aver_sent)
print('Variance score: %.2f' % r2_score(Daily_gross, gross_pred))


regr.fit(Rank, Daily_gross)
gross_pred = regr.predict(Rank)
print('Variance score: %.2f' % r2_score(Daily_gross, gross_pred))

regr.fit(Fav, Daily_gross)
gross_pred = regr.predict(Fav)
print('Variance score: %.2f' % r2_score(Daily_gross, gross_pred))

regr.fit(Retweett, Daily_gross)
gross_pred = regr.predict(Retweett)
print('Variance score: %.2f' % r2_score(Daily_gross, gross_pred))

regr.fit(Daily_theatre, Daily_gross)
gross_pred = regr.predict(Daily_theatre)
print('Variance score: %.2f' % r2_score(Daily_gross, gross_pred))

Use = pd.concat([Aver_sent, Rank, Fav, Retweett, Daily_theatre], axis=1, 
                join_axes=[Aver_sent.index])


regr.fit(Use, Daily_gross)
gross_pred = regr.predict(Use)
print('Variance score: %.2f' % r2_score(Daily_gross, gross_pred))


MLPClassifier(solver='lbfgs', hidden_layer_sizes=(15,), random_state=1,alpha=1e-5)

###PIPELINED RANDOM FOREST FEATURE
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

text_clfPipe = Pipeline([
                      ('clf-rf', RandomForestClassifier(max_depth=2, random_state=0)),
])

text_clf = text_clfPipe.fit(Use, Daily_gross)
predicted = text_clf.predict(Use)
print('Variance score: %.2f' % r2_score(Daily_gross, predicted))
acc = np.mean(predicted == twenty_test.target)



