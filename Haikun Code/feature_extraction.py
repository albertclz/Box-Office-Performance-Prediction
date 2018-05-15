#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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

#words = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_13_revised.csv')


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
df = df[['word','freq']]
#df = df.drop(['movie','see','watching','watched','wanna'])
tuples = [tuple(x) for x in df.values]


#create wordcloud and plot final pictures
wordcloud = WordCloud(background_color="white",width=1000,height=860,margin=2).generate_from_frequencies(tuples)

wordcloud.to_file('/Users/haikundu/Desktop/COMS6998/finalproject/whole_page.png')
import matplotlib.pyplot as plt

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
