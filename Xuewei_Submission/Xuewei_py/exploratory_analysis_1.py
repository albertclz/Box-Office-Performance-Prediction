#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:07:44 2017

@author: duxuewei
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:47:02 2017

@author: duxuewei
"""

#Exploratory data analysis

#Citation: function genCorpus
#Citation: Regarding wordcloud:https://github.com/amueller/word_cloud


import pandas as pd
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
import matplotlib.pyplot as plt

thePath = '/Users/duxuewei/Desktop/projectds/project/'

movie_tokens = ["happy", "death", "day"]

raw_df = pd.read_csv(thePath + "10_15.csv", sep = ";", error_bad_lines = False, na_filter = True)
raw_df = raw_df.dropna(subset = ["text"])

print raw_df.columns

#raw_df = raw_df[["date", "retweets", "favorites", "text", "mentions", "hashtags"]]

tweet_text = raw_df.text


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
    #tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = [word for word in tokens if word not in movie_tokens]#Removes movie name from tokens
    tokens = " ".join(tokens) #need to pass string seperated by spaces       

    return tokens

tokenized_tweets = list()
for tweet in tweet_text:
     tokenized_tweets.append(genCorpus(tweet))
    
    
###############Get wordcloud 
#conda install -c https://conda.anaconda.org/amueller wordcloud
from wordcloud import WordCloud

# Read the whole text.
text = " ".join(tokenized_tweets)

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
#import matplotlib.pyplot as plt
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



#How many words are ther in each tweet?
tweet_sizes = list()
for tweet in tweet_text:
    tweet_sizes.append(len(tweet.split()))
    #try: 
    #    len(tweet.split())
    #except: 
    #    print tweet
    #    pass
plt.plot(tweet_sizes)
plt.show()
plt.close()
plt.hist(tweet_sizes, 15, facecolor='blue', alpha=0.75)
plt.xlabel('Number of words')
plt.ylabel('Frequency per additional word')
plt.title(r'$\mathrm{Histogram\ of\ Number\ of\ words\ in\ a\ tweet}$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
plt.close()

#Show top 20 unique hashtags
from collections import Counter
unique_hashtags = raw_df.hashtags.unique()
#print(len(unique_hashtags))
counts = Counter(raw_df.hashtags)
counts.most_common(15)


#Convert time string to datetime object
from datetime import datetime
datetime_object = list()
for raw_date in raw_df.date:
    datetime_object.append(datetime.strptime(raw_date, '%Y-%m-%d %H:%M'))
raw_df.date = pd.Series(datetime_object)
#df.set_index('Date_Time').groupby(pd.TimeGrouper('D')).mean().dropna()
hours = list()
for x in raw_df.date:
    hours.append(x.hour)
raw_df["hour"] = pd.Series(hours)
hour_based_df = raw_df.groupby("hour").size()
plt.plot(hour_based_df)
plt.show()
plt.close()


