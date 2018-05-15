#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:22:58 2017

@author: haikundu
"""
import got
import datetime
import pandas as pd

movie_list = ['Aloha','Project Almanac','The Gallows','Ex Machina','Run All Night']
open_date = ['2015-05-29','2015-01-30','2015-07-10','2015-04-10','2015-03-13']

num_of_movie = len(movie_list)
for cnt in range(0,num_of_movie):
    time = 0
    while time < 14:
        if time == 0:
            cur_date = datetime.datetime.strptime(open_date[cnt],'%Y-%m-%d')
        next_date = cur_date + datetime.timedelta(days=1)
        start = cur_date.strftime('%Y-%m-%d')
        end = next_date.strftime('%Y-%m-%d')
                
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(movie_list[cnt]).setSince(start).setUntil(end).setMaxTweets(100000)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        body = []
        date = []
        retweets = []
        favorites = []
        mentions = []
        hashtags = []
        geo = []
        for tweet in tweets:	  
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
        df.to_csv(movie_list[cnt]+ start + '.csv', index = False, encoding = 'utf-8')
        print movie_list[cnt]+ start
        time += 1
        cur_date = next_date