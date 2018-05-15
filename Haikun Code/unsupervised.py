#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:58:52 2017

@author: haikundu
"""

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import csv

stopwords = nltk.corpus.stopwords.words('english')


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) \
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) \
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

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


'''
# Data for Cornell Movie Review
thePath = '/Users/haikundu/Desktop/COMS6998/finalproject/test/'
finalWords = list()
cnt = 0
for file in os.listdir(thePath):
    if file.endswith('.txt'):
        try:
            cnt = cnt + 1
            f = open(thePath + file, "r")
            lines = f.readlines()
            lines = [text.strip() for text in lines]
            lines = " ".join(lines)
            test = ' '.join(tokenize_and_stem(lines))
            finalWords.append(test)
        except:
            pass
'''


#Data of Happy Death Day
'''
thePath = '/Users/haikundu/Desktop/COMS6998/finalproject/row_data/'
whole_page = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/row_data/10_13New.csv')
for date in range(14,23):
    try:
        date = str(date)
        files = pd.read_csv(thePath + '10_' + date + 'New.csv')
        whole_page = pd.concat([whole_page,files])
    except:
        print date
        pass
'''

whole_page = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/output.csv')
blank = whole_page['text'].notnull()
filtered = whole_page[blank]
sent = filtered['sent'].tolist()

'''
#filter 'happy death day'
lines = whole_page.dropna().values.tolist()
final1 = []
final2 = []
final3 = []
for text in lines:
    final1.extend(text)
    
for texts in final1:
    filtered = texts.replace('happy death day ','')
    final2.append(filtered)

for textss in final2:
    filtereds = textss.replace('happy death day','')
    if filtereds:
        final3.append(filtereds)
'''
#whole page per day
finalWords = filtered['text'].tolist()

'''
#Data for Wine
row_page = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/wine/wine.csv')
df = pd.DataFrame(row_page)
df = df[df.duplicated('description',keep=False)]
df = df[:10000]
finalWords = df['description'].tolist()
'''

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000,stop_words='english',tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(finalWords)

terms = tfidf_vectorizer.get_feature_names()


#KMeans Cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(tfidf_matrix)
predictions = kmeans.predict(tfidf_matrix)
text_type = list(predictions)
filtered['text_type'] = text_type

'''
#determine the number of types

K = range(1, 10)
X = tfidf_matrix
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    print kmeans.inertia_

import matplotlib.pyplot as plt
xx = [1,2,3,4,5,6,7,8,9]
yy = [37326.2092595
,35561.1667979
,34918.420741
,34439.6345442
,33448.9257949
,33324.8354842
,33180.8241576
,32416.4496208
,32321.5308102]
plt.plot(xx,yy)
'''



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
    if text_type[i] == 0:
        if sent[i] == 'Pos':
            type0_pos = pd.concat([type0_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type0_neg = pd.concat([type0_neg,filtered[i:i+1]])

    elif text_type[i] == 1:
        if sent[i] == 'Pos':
            type1_pos = pd.concat([type1_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type1_neg = pd.concat([type1_neg,filtered[i:i+1]])
            
    elif text_type[i] == 2:
        if sent[i] == 'Pos':
            type2_pos = pd.concat([type2_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type2_neg = pd.concat([type2_neg,filtered[i:i+1]])
            
    elif text_type[i] == 3:
        if sent[i] == 'Pos':
            type3_pos = pd.concat([type3_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type3_neg = pd.concat([type3_neg,filtered[i:i+1]])
    else:
        if sent[i] == 'Pos':
            type4_pos = pd.concat([type4_pos,filtered[i:i+1]])
        elif sent[i] == 'Neg':
            type4_neg = pd.concat([type4_neg,filtered[i:i+1]])

type1_pos = type1_pos.sort_values(by='ranking',ascending=0)
type2_pos = type2_pos.sort_values(by='ranking',ascending=0)
type3_pos = type3_pos.sort_values(by='ranking',ascending=0)
type0_pos = type4_pos.sort_values(by='ranking',ascending=0)
type4_pos = type0_pos.sort_values(by='ranking',ascending=0)

type1_neg = type1_neg.sort_values(by='ranking',ascending=0)
type2_neg = type2_neg.sort_values(by='ranking',ascending=0)
type3_neg = type3_neg.sort_values(by='ranking',ascending=0)
type0_neg = type4_neg.sort_values(by='ranking',ascending=0)
type4_neg = type0_neg.sort_values(by='ranking',ascending=0)
'''        
t1 = pd.DataFrame(type1).to_csv('type1.csv')
t2 = pd.DataFrame(type2).to_csv('type2.csv')
t3 = pd.DataFrame(type3).to_csv('type3.csv')
t4 = pd.DataFrame(type4).to_csv('type4.csv')
t0 = pd.DataFrame(type0).to_csv('type0.csv')
'''