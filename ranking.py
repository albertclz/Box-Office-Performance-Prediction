#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:24:27 2017

@author: haikundu
"""

import pandas as pd

whole_page_sent = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/ranking_whole_new.csv')

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

#compute ranings
retweets = whole_page['retweets'].tolist()
retweets_mean = whole_page['retweets'].mean()
retweets_var = whole_page['retweets'].var()
retweets = [(row-retweets_mean)/retweets_var for row in retweets]
    

favorite = whole_page['favorites'].tolist()
favorite_mean = whole_page['favorites'].mean()
favorite_var = whole_page['favorites'].var()
favorite = [(row-favorite_mean)/favorite_var for row in favorite]

texts = whole_page_sent['0'].tolist()
length = [len(text) for text in texts]
whole_page['length'] = length
length_mean = whole_page['length'].mean()
length_var = whole_page['length'].var()
length = [(row-length_mean)/length_var for row in length]

ranking = []
for num in range(len(whole_page)):
    ranking.append(retweets[num]+favorite[num]+length[num])

# estimate the semtiment type for each tweet 
Neg = whole_page_sent['Negative_Score'].tolist()
Pos = whole_page_sent['Positive_Score'].tolist()
Neu = whole_page_sent['Neutral_Score'].tolist()

polarity = list()
for i in range(0,len(Neg)):
    if Neg[i] > Pos[i]:
        if Neg[i] > Neu[i]:
            polarity.append('Neg')
        else:
            polarity.append('Neu')
    else:
        if Pos[i] > Neu[i]:
            polarity.append('Pos')
        else:
            polarity.append('Neu')

#combine final output
output = pd.DataFrame()
output['text'] = whole_page['text']
output['sent'] = polarity
output['ranking'] = ranking


output.to_csv('output.csv', index = False, encoding = 'utf-8')
