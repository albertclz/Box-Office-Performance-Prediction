#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:30:42 2017

@author: haikundu
"""

import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

thePath = '/Users/haikundu/Desktop/COMS6998/finalproject/revised/'
whole_page = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_13_revised.csv')
for date in range(14,23):
    try:
        date = str(date)
        files = pd.read_csv(thePath + '10_' + date + '_revised.csv')
        whole_page = pd.concat([whole_page,files])
    except:
        print date
        pass

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

#whole page per day
finalWords = ' '.join(final3)

#TFIDF methods
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,3))
tdm = pd.DataFrame(vectorizer.fit_transform(final3).toarray())
tdm.columns=vectorizer.get_feature_names()

test = vectorizer.transform([finalWords])
freq = test.toarray()

result = pd.DataFrame(freq)
result.columns = tdm.columns
result = result.transpose()
result.columns = ['freq']
result['word'] = tdm.columns.tolist()
df = result.sort_values(by='freq',ascending=0)
finaldf = df[['word','freq']]


for date in range(13,23):
    date = str(date)
    files = pd.read_csv(thePath + '10_' + date + '_revised.csv')
    
    #filter 'happy death day'
    lines = files.dropna().values.tolist()
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
    
    #whole page per day
    finalWords = ' '.join(final3)
    
    #TFIDF methods
    vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,3))
    tdm = pd.DataFrame(vectorizer.fit_transform(final3).toarray())
    tdm.columns=vectorizer.get_feature_names()
    
    test = vectorizer.transform([finalWords])
    freq = test.toarray()
    
    result = pd.DataFrame(freq)
    result.columns = tdm.columns
    result = result.transpose()
    result.columns = ['freq'+date]
    result['word'] = tdm.columns.tolist()
    finaldf = pd.merge(finaldf,result)
    
top20 = finaldf.head(21)
top20 = top20.drop([12])
wholetop20 = top20.pop('freq')
from pandas.tools.plotting import parallel_coordinates
graph = parallel_coordinates(top20, 'word')
graph.to_file('/Users/haikundu/Desktop/COMS6998/finalproject/top20_trend.png')




