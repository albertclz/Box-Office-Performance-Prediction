# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:16:31 2017

@author: Jack. Wang
"""
import os
os.chdir('D:\Columbia-Course\CS-6998\GetOldTweets-python-master')
import got
import datetime
import pandas as pd
import csv

"""
The movie collector py file is using the GetOldTweets python file which is available at the link below,
please feel free to take a look:

https://github.com/Jefferson-Henrique/GetOldTweets-python

Before use GetOldTweets, please download the whole folder and take a look of the README.
To make sure everything is setted up porperly. Make sure this python file is putted
under the same directory as the GetOldTweet file. Thank you.
"""

os.chdir('D:\Columbia-Course\CS-6998\Movie_Data')
os.getcwd()

movie_list = ['Happy Death Day',
              'Atomic Blonde','American Made','The Dark Tower', 'A Bad Moms Christmas', 'Atomic Blonde','American Made','Wonder',
              'Snatched','The Great Wall','Going in Style','All Eyez on Me','47 Meters Down','The Big Sick','Ghost in the Shell','Jigsaw',
              'American Assassin','The Foreigner','Wind River','Monster Trucks','Geostorm','	Fist Fight','Kidnap','Rings','Logan Lucky','Home Again',
              'The Bye Bye Man','Victoria and Abdul']

open_date = ['10-13',
             '07-28','09-29',
             '08-04', '11-01', '07-28','9-29','11-17','5-12','02-17','04-07','06-16','06-16','06-23','03-31','10-27','09-15',
             '10-13','08-04','01-13','10-20','02-17','08-04','02-03','08-18','09-08','01-13','09-22']

for i in range(0, len(open_date)):
    open_date[i] = '2017-'+ open_date[i]

num_of_movie = len(movie_list)
for cnt in range(0,num_of_movie):
    time = 0
    while time < 14:
        try:
            startrunning = datetime.datetime.now()
            if time == 0:
                cur_date = datetime.datetime.strptime(open_date[cnt],'%Y-%m-%d')
            next_date = cur_date + datetime.timedelta(days=1)
            start = cur_date.strftime('%Y-%m-%d')
            end = next_date.strftime('%Y-%m-%d')
            #tweetCriteria = got.manager.TweetCriteria().setQuerySearch(movie_list[cnt]).setSince(start).setMaxTweets(20000)
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(movie_list[cnt]).setSince(start).setUntil(end).setMaxTweets(20000)
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)
            body = []
            date = []
            retweets = []
            favorites = []
            mentions = []
            hashtags = []
            geo = []
            for tweet in tweets:
                #print(tweet.text)
                body.append(tweet.text)
                date.append(tweet.date)
                retweets.append(tweet.retweets)
                favorites.append(tweet.favorites)
                mentions.append(tweet.mentions)
                hashtags.append(tweet.hashtags)
                geo.append(tweet.geo)
            df = pd.DataFrame(columns=['body','date','retweets','favorites','mentions','hashtags','geo'])
            df['body']=body
            df['date']=date
            df['retweets']=retweets
            df['favorites']=favorites
            df['mentions']=mentions
            df['hashtags']=hashtags
            df['geo']=geo
            df.to_csv(movie_list[cnt]+ '-' + start + '.csv', index = False, encoding = 'utf-8')
            print movie_list[cnt]+ start
            print(datetime.datetime.now() - startrunning)
            time += 1
            cur_date = next_date
        except:
            print time
            pass
