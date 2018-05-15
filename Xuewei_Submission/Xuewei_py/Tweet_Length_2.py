#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 03:10:52 2017

@author: duxuewei
"""

#Exploratory data analysis - histogram of tweet length

import pandas as pd
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
import matplotlib.pyplot as plt

thePath = '/Users/duxuewei/Desktop/projectds/project/'

#Read in data
tweet_text = pd.Series()
for i in range(13,23):
    raw_df = pd.read_csv(thePath + "10_" + str(i) + ".csv", sep = ";", error_bad_lines = False, na_filter = True)
    raw_df = raw_df.dropna(subset = ["text"])
    #print raw_df.text
    tweet_text = tweet_text.append(raw_df.text, ignore_index = True)

###How many words are ther in each tweet?
tweet_sizes = list()
for tweet in tweet_text:
    tweet_sizes.append(len(tweet.split()))
    #try: 
    #    len(tweet.split())
    #except: 
    #    print tweet
    #    pass

"""
plt.plot(tweet_sizes)
plt.show()
plt.close()
"""
#Ploting a histogram for tweet length(size)
plt.hist(tweet_sizes, 15, facecolor='blue', alpha=0.75)
plt.xlabel('Number of words')
plt.ylabel('Frequency per additional word')
plt.title(r'$\mathrm{Histogram\ of\ Number\ of\ words\ in\ a\ tweet}$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
plt.close()


#Save data to csv file
df_tweet_length = pd.DataFrame({'tweet_sizes': tweet_sizes})
df_tweet_length.to_csv(thePath + "tweet_sizes.csv", index = False)
