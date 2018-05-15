# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:34:03 2017

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


os.getcwd()
os.chdir('D:\Columbia-Course\CS-6998\Project\Review_Sets')

corpus = pd.read_csv('10_24.csv', 
                    sep=";")


corpus1 = pd.read_csv('10_24.csv', 
                    sep=";")


corpus = corpus[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus1 = corpus1[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]

corpus['Sentiment']= 'NA'; corpus['Sentiment_Score']= 'NA'
corpus1['Compound_Score']= 'NA'; corpus1['Negative_Score']= 'NA'; corpus1['Positive_Score']= 'NA'; corpus1['Neutral_Score']= 'NA'

corpus.head(5)


"""
Functions defined
"""
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

def get_tweet_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def sentiment_score(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

sid = SentimentIntensityAnalyzer()
def SID_Analyzer(text):
    result =  sid.polarity_scores(text)
    return result


"""
Method 1 - All Info append into corpus Dataframe, using TextBlob classifier
"""
cnt = 0
for i in corpus['text']:
    info = genCorpus(i)
    if info != '':
        corpus['text'][cnt] = info
    else:
        corpus['text'][cnt] = 'neutral'
    corpus['Sentiment'][cnt] = get_tweet_sentiment(corpus['text'][cnt])
    corpus['Sentiment_Score'][cnt] = sentiment_score(corpus['text'][cnt])
    cnt +=1

"""
Method 2 - Have different score result from Method 1, but using standard NLTK classifier
"""

"""
Small Sample Example
"""
small_set = corpus[1:11]
sid = SentimentIntensityAnalyzer()
for i in small_set['text']:
    ss = sid.polarity_scores(i)
    print ss

s1 = sid.polarity_scores(corpus['text'][11])
s1['compound']

"""
Complete Data Sentiment Analysis
"""
cnt = 0
for i in corpus1['text']:
    info = genCorpus(i)
    if info != '':
        corpus1['text'][cnt] = info
    else:
        corpus1['text'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus['text'][cnt])
    corpus1['Compound_Score'][cnt] = compound['compound']
    corpus1['Negative_Score'][cnt] = compound['neg']
    corpus1['Positive_Score'][cnt] = compound['pos']
    corpus1['Neutral_Score'][cnt] = compound['neu']
    cnt +=1



"""
Conduct full sentiment analysis for all tweet from 10_13 to 10_24
Using method 2 - since it provides more complete sense of what is going on
"""
corpus_13 = pd.read_csv('10_13.csv', sep=";",error_bad_lines = False)
corpus_14 = pd.read_csv('10_14.csv', sep=";",error_bad_lines = False)
corpus_15 = pd.read_csv('10_15.csv', sep=";",error_bad_lines = False)
corpus_16 = pd.read_csv('10_16.csv', sep=";",error_bad_lines = False)
corpus_17 = pd.read_csv('10_17.csv', sep=";",error_bad_lines = False)
corpus_18 = pd.read_csv('10_18.csv', sep=";",error_bad_lines = False)
corpus_19 = pd.read_csv('10_19.csv', sep=";",error_bad_lines = False)
corpus_20 = pd.read_csv('10_20.csv', sep=";",error_bad_lines = False)
corpus_21 = pd.read_csv('10_21.csv', sep=";",error_bad_lines = False)
corpus_22 = pd.read_csv('10_22.csv', sep=";",error_bad_lines = False)

corpus_13 = corpus_13[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_14 = corpus_14[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_15 = corpus_15[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_16 = corpus_16[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_17 = corpus_17[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_18 = corpus_18[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_19 = corpus_19[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_20 = corpus_20[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_21 = corpus_21[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]
corpus_22 = corpus_22[['username', 'date', 'retweets','favorites','text','mentions','hashtags','id']]

corpus_13['Compound_Score']= 'NA'; corpus_13['Negative_Score']= 'NA'; corpus_13['Positive_Score']= 'NA'; corpus_13['Neutral_Score']= 'NA'
corpus_14['Compound_Score']= 'NA'; corpus_14['Negative_Score']= 'NA'; corpus_14['Positive_Score']= 'NA'; corpus_14['Neutral_Score']= 'NA'
corpus_15['Compound_Score']= 'NA'; corpus_15['Negative_Score']= 'NA'; corpus_15['Positive_Score']= 'NA'; corpus_15['Neutral_Score']= 'NA'
corpus_16['Compound_Score']= 'NA'; corpus_16['Negative_Score']= 'NA'; corpus_16['Positive_Score']= 'NA'; corpus_16['Neutral_Score']= 'NA'
corpus_17['Compound_Score']= 'NA'; corpus_17['Negative_Score']= 'NA'; corpus_17['Positive_Score']= 'NA'; corpus_17['Neutral_Score']= 'NA'
corpus_18['Compound_Score']= 'NA'; corpus_18['Negative_Score']= 'NA'; corpus_18['Positive_Score']= 'NA'; corpus_18['Neutral_Score']= 'NA'
corpus_19['Compound_Score']= 'NA'; corpus_19['Negative_Score']= 'NA'; corpus_19['Positive_Score']= 'NA'; corpus_19['Neutral_Score']= 'NA'
corpus_20['Compound_Score']= 'NA'; corpus_20['Negative_Score']= 'NA'; corpus_20['Positive_Score']= 'NA'; corpus_20['Neutral_Score']= 'NA'
corpus_21['Compound_Score']= 'NA'; corpus_21['Negative_Score']= 'NA'; corpus_21['Positive_Score']= 'NA'; corpus_21['Neutral_Score']= 'NA'
corpus_22['Compound_Score']= 'NA'; corpus_22['Negative_Score']= 'NA'; corpus_22['Positive_Score']= 'NA'; corpus_22['Neutral_Score']= 'NA'
corpus_23['Compound_Score']= 'NA'; corpus_23['Negative_Score']= 'NA'; corpus_23['Positive_Score']= 'NA'; corpus_23['Neutral_Score']= 'NA'
corpus_24['Compound_Score']= 'NA'; corpus_24['Negative_Score']= 'NA'; corpus_24['Positive_Score']= 'NA'; corpus_24['Neutral_Score']= 'NA'
corpus_25['Compound_Score']= 'NA'; corpus_25['Negative_Score']= 'NA'; corpus_25['Positive_Score']= 'NA'; corpus_25['Neutral_Score']= 'NA'
corpus_26['Compound_Score']= 'NA'; corpus_26['Negative_Score']= 'NA'; corpus_26['Positive_Score']= 'NA'; corpus_26['Neutral_Score']= 'NA'

corpus_13 = corpus_13[corpus_13['body'].notnull()]
corpus_14 = corpus_14[corpus_14['body'].notnull()]
corpus_15 = corpus_15[corpus_15['body'].notnull()]
corpus_16 = corpus_16[corpus_16['body'].notnull()]
corpus_17 = corpus_17[corpus_17['body'].notnull()]
corpus_18 = corpus_18[corpus_18['body'].notnull()]
corpus_19 = corpus_19[corpus_19['body'].notnull()]
corpus_20 = corpus_20[corpus_20['body'].notnull()]
corpus_21 = corpus_21[corpus_21['body'].notnull()]
corpus_22 = corpus_22[corpus_22['body'].notnull()]
corpus_23 = corpus_23[corpus_23['body'].notnull()]
corpus_24 = corpus_24[corpus_24['body'].notnull()]
corpus_25 = corpus_25[corpus_25['body'].notnull()]
corpus_26 = corpus_26[corpus_26['body'].notnull()]

corpus_13.index = range(0,len(corpus_13['body']))
corpus_14.index = range(0,len(corpus_14['body']))
corpus_15.index = range(0,len(corpus_15['body']))
corpus_16.index = range(0,len(corpus_16['body']))
corpus_17.index = range(0,len(corpus_17['body']))
corpus_18.index = range(0,len(corpus_18['body']))
corpus_19.index = range(0,len(corpus_19['body']))
corpus_20.index = range(0,len(corpus_20['body']))
corpus_21.index = range(0,len(corpus_21['body']))
corpus_22.index = range(0,len(corpus_22['body']))
corpus_23.index = range(0,len(corpus_23['body']))
corpus_24.index = range(0,len(corpus_24['body']))
corpus_25.index = range(0,len(corpus_25['body']))
corpus_26.index = range(0,len(corpus_26['body']))

cnt=0
for i in corpus_13['body']:
    info = genCorpus(i)
    if info != '':
        corpus_13['body'][cnt] = info
    else:
        corpus_13['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_13['body'][cnt])
    corpus_13['Compound_Score'][cnt] = compound['compound']
    corpus_13['Negative_Score'][cnt] = compound['neg']
    corpus_13['Positive_Score'][cnt] = compound['pos']
    corpus_13['Neutral_Score'][cnt] = compound['neu']
    cnt +=1

cnt=0
for i in corpus_14['body']:
    info = genCorpus(i)
    if info != '':
        corpus_14['body'][cnt] = info
    else:
        corpus_14['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_14['body'][cnt])
    corpus_14['Compound_Score'][cnt] = compound['compound']
    corpus_14['Negative_Score'][cnt] = compound['neg']
    corpus_14['Positive_Score'][cnt] = compound['pos']
    corpus_14['Neutral_Score'][cnt] = compound['neu']
    cnt +=1

cnt=0
for i in corpus_15['body']:
    info = genCorpus(i)
    if info != '':
        corpus_15['body'][cnt] = info
    else:
        corpus_15['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_15['body'][cnt])
    corpus_15['Compound_Score'][cnt] = compound['compound']
    corpus_15['Negative_Score'][cnt] = compound['neg']
    corpus_15['Positive_Score'][cnt] = compound['pos']
    corpus_15['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_16['body']:
    info = genCorpus(i)
    if info != '':
        corpus_16['body'][cnt] = info
    else:
        corpus_16['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_16['body'][cnt])
    corpus_16['Compound_Score'][cnt] = compound['compound']
    corpus_16['Negative_Score'][cnt] = compound['neg']
    corpus_16['Positive_Score'][cnt] = compound['pos']
    corpus_16['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_17['body']:
    info = genCorpus(i)
    if info != '':
        corpus_17['body'][cnt] = info
    else:
        corpus_17['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_17['body'][cnt])
    corpus_17['Compound_Score'][cnt] = compound['compound']
    corpus_17['Negative_Score'][cnt] = compound['neg']
    corpus_17['Positive_Score'][cnt] = compound['pos']
    corpus_17['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_18['body']:
    info = genCorpus(i)
    if info != '':
        corpus_18['body'][cnt] = info
    else:
        corpus_18['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_18['body'][cnt])
    corpus_18['Compound_Score'][cnt] = compound['compound']
    corpus_18['Negative_Score'][cnt] = compound['neg']
    corpus_18['Positive_Score'][cnt] = compound['pos']
    corpus_18['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_19['body']:
    info = genCorpus(i)
    if info != '':
        corpus_19['body'][cnt] = info
    else:
        corpus_19['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_19['body'][cnt])
    corpus_19['Compound_Score'][cnt] = compound['compound']
    corpus_19['Negative_Score'][cnt] = compound['neg']
    corpus_19['Positive_Score'][cnt] = compound['pos']
    corpus_19['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_20['body']:
    info = genCorpus(i)
    if info != '':
        corpus_20['body'][cnt] = info
    else:
        corpus_20['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_20['body'][cnt])
    corpus_20['Compound_Score'][cnt] = compound['compound']
    corpus_20['Negative_Score'][cnt] = compound['neg']
    corpus_20['Positive_Score'][cnt] = compound['pos']
    corpus_20['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_21['body']:
    info = genCorpus(i)
    if info != '':
        corpus_21['body'][cnt] = info
    else:
        corpus_21['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_21['body'][cnt])
    corpus_21['Compound_Score'][cnt] = compound['compound']
    corpus_21['Negative_Score'][cnt] = compound['neg']
    corpus_21['Positive_Score'][cnt] = compound['pos']
    corpus_21['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
    
cnt=0
for i in corpus_22['body']:
    info = genCorpus(i)
    if info != '':
        corpus_22['body'][cnt] = info
    else:
        corpus_22['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_22['body'][cnt])
    corpus_22['Compound_Score'][cnt] = compound['compound']
    corpus_22['Negative_Score'][cnt] = compound['neg']
    corpus_22['Positive_Score'][cnt] = compound['pos']
    corpus_22['Neutral_Score'][cnt] = compound['neu']
    cnt +=1

cnt=0
for i in corpus_23['body']:
    info = genCorpus(i)
    if info != '':
        corpus_23['body'][cnt] = info
    else:
        corpus_23['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_23['body'][cnt])
    corpus_23['Compound_Score'][cnt] = compound['compound']
    corpus_23['Negative_Score'][cnt] = compound['neg']
    corpus_23['Positive_Score'][cnt] = compound['pos']
    corpus_23['Neutral_Score'][cnt] = compound['neu']
    cnt +=1

cnt=0
for i in corpus_24['body']:
    info = genCorpus(i)
    if info != '':
        corpus_24['body'][cnt] = info
    else:
        corpus_24['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_24['body'][cnt])
    corpus_24['Compound_Score'][cnt] = compound['compound']
    corpus_24['Negative_Score'][cnt] = compound['neg']
    corpus_24['Positive_Score'][cnt] = compound['pos']
    corpus_24['Neutral_Score'][cnt] = compound['neu']
    cnt +=1

cnt=0
for i in corpus_25['body']:
    info = genCorpus(i)
    if info != '':
        corpus_25['body'][cnt] = info
    else:
        corpus_25['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_25['body'][cnt])
    corpus_25['Compound_Score'][cnt] = compound['compound']
    corpus_25['Negative_Score'][cnt] = compound['neg']
    corpus_25['Positive_Score'][cnt] = compound['pos']
    corpus_25['Neutral_Score'][cnt] = compound['neu']
    cnt +=1

cnt=0
for i in corpus_26['body']:
    info = genCorpus(i)
    if info != '':
        corpus_26['body'][cnt] = info
    else:
        corpus_26['body'][cnt] = 'neutral'
    compound = SID_Analyzer(corpus_26['body'][cnt])
    corpus_26['Compound_Score'][cnt] = compound['compound']
    corpus_26['Negative_Score'][cnt] = compound['neg']
    corpus_26['Positive_Score'][cnt] = compound['pos']
    corpus_26['Neutral_Score'][cnt] = compound['neu']
    cnt +=1
"""
Reorganize dataframe

Change date from str to timestamp
"""
matplotlib.style.use('ggplot')
corpus_13['date'] = pd.to_datetime(corpus_13['date'])
corpus_14['date'] = pd.to_datetime(corpus_14['date'])
corpus_15['date'] = pd.to_datetime(corpus_15['date'])
corpus_16['date'] = pd.to_datetime(corpus_16['date'])
corpus_17['date'] = pd.to_datetime(corpus_17['date'])
corpus_18['date'] = pd.to_datetime(corpus_18['date'])
corpus_19['date'] = pd.to_datetime(corpus_19['date'])
corpus_20['date'] = pd.to_datetime(corpus_20['date'])
corpus_21['date'] = pd.to_datetime(corpus_21['date'])
corpus_22['date'] = pd.to_datetime(corpus_22['date'])
corpus_23['date'] = pd.to_datetime(corpus_23['date'])
corpus_24['date'] = pd.to_datetime(corpus_24['date'])
corpus_25['date'] = pd.to_datetime(corpus_25['date'])
corpus_26['date'] = pd.to_datetime(corpus_26['date'])

exam['date']=exam['date'].map(lambda x: x.day+1)

exam['Month'][1]

exam = corpus_13
exam['date'] = pd.to_datetime(exam['date'])
exam['Hour'] = pd.DatetimeIndex(exam['date']).hour
exam['Minute'] = pd.DatetimeIndex(exam['date']).minute

exam[['Compound_Score']].plot.hist(alpha=0.9,bins=30)
exam[['Negative_Score']].plot.hist(alpha=0.9)
exam[['Positive_Score']].plot.hist(alpha=0.9)
exam[['Neutral_Score']].plot.hist(alpha=0.9)

exam1 = exam[['Compound_Score','Positive_Score','Negative_Score','Neutral_Score']]

exam[['Compound_Score','Positive_Score','Negative_Score','Neutral_Score']].plot.hist(alpha=0.6, bins = 30)

exam1.plot.box()

tlen = pd.Series(data = exam['Compound_Score'].values, index = exam['Minute'])
tlen.plot(figsize = (10,4),color = 'r', lw=0.5)


"""
Using NLTK-Trainer to train classifier

python train_classifier.py movie_reviews --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 --min_score 3

"""
NaiveBayes_classifier = pickle.load(open("D:\Columbia-Course\CS-6998\Project\movie_reviews_NaiveBayes.pickle"))


"""
Output result as CSV file
"""
os.getcwd()
os.chdir('D:\Columbia-Course\CS-6998\Project')

corpus.to_csv('corpus.csv')
corpus1.to_csv('corpus1.csv')
corpus_13.to_csv('corpus_13.csv')
corpus_14.to_csv('corpus_14.csv')
corpus_15.to_csv('corpus_15.csv')
corpus_16.to_csv('corpus_16.csv')
corpus_17.to_csv('corpus_17.csv')
corpus_18.to_csv('corpus_18.csv')
corpus_19.to_csv('corpus_19.csv')
corpus_20.to_csv('corpus_20.csv')
corpus_21.to_csv('corpus_21.csv')
corpus_22.to_csv('corpus_22.csv')


"""
Descriptive Statistics
"""
corpus['Sentiment'].count()
corpus.groupby('Sentiment').size()
corpus.groupby('date').size()


type(corpus['date'][1])

"""
Overall Reading in
"""
os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\Deadpool')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'Deadpool')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'Deadpool')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'Deadpool')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'Deadpool')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'Deadpool')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'Deadpool')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'Deadpool')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'Deadpool')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'Deadpool')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'Deadpool')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'Deadpool')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'Deadpool')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'Deadpool')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'Deadpool')[13])
    cnt +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\The_Secret_Life_of_Pets')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'The_Secret_Life_of_Pets')[13])
    cnt +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data\')
thePath = 'D:\\Columbia-Course\\CS-6998\\Movie_Data\\'
cnt = 0
while cnt < 1:
    corpus_13 = pd.read_csv(os.listdir(thePath+'')[0])
    corpus_14 = pd.read_csv(os.listdir(thePath+'')[1])
    corpus_15 = pd.read_csv(os.listdir(thePath+'')[2])
    corpus_16 = pd.read_csv(os.listdir(thePath+'')[3])
    corpus_17 = pd.read_csv(os.listdir(thePath+'')[4])
    corpus_18 = pd.read_csv(os.listdir(thePath+'')[5])
    corpus_19 = pd.read_csv(os.listdir(thePath+'')[6])
    corpus_20 = pd.read_csv(os.listdir(thePath+'')[7])
    corpus_21 = pd.read_csv(os.listdir(thePath+'')[8])
    corpus_22 = pd.read_csv(os.listdir(thePath+'')[9])
    corpus_23 = pd.read_csv(os.listdir(thePath+'')[10])
    corpus_24 = pd.read_csv(os.listdir(thePath+'')[11])
    corpus_25 = pd.read_csv(os.listdir(thePath+'')[12])
    corpus_26 = pd.read_csv(os.listdir(thePath+'')[13])
    i +=1



















