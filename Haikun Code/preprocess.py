#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re


data = pd.read_csv("/Users/haikundu/Desktop/COMS6998/finalproject/GetOldTweet/10_22.csv",sep=';',error_bad_lines = False)

'''
#merge all tweets
thePath = '/Users/haikundu/Desktop/COMS6998/finalproject/GetOldTweet/data/'
whole_page = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/GetOldTweet/data/10_13.csv',sep=';',error_bad_lines = False)
for date in range(14,23):
    try:
        date = str(date)
        files = pd.read_csv(thePath + '10_' + date + '.csv',sep=';',error_bad_lines = False)
        whole_page = pd.concat([whole_page,files])
    except:
        print date
        pass
'''

def genCorpus(theText):
    #set dictionaries
    stopWords = set(stopwords.words('english'))
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
    
    #pre-processing
    theText = theText.split(' ')
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words
    #tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces       

    return tokens

Text = data["text"].tolist()
body = []
for words in Text:
    try:
        body.append(genCorpus(words))
    except:
        pass
    
df = pd.DataFrame(body)
df.to_csv('10_22_revised.csv', index = False, encoding = 'utf-8')