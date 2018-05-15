#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:10:33 2017

@author: duxuewei
"""

import os
os.chdir('/Users/duxuewei/Desktop/projectds/project/GetOldTweets-python-master/')
import got
import datetime
import pandas as pd



import pandas as pd
import numpy as np
from sklearn import feature_extraction
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import numpy
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import decomposition
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
from sklearn.externals import joblib
import sys
import nltk
from nltk.corpus import stopwords
import re
import os

import math




thePathLut = '/Users/duxuewei/Desktop/projectds/project/'

os.chdir('/Users/duxuewei/Desktop/projectds/project/supervised_learning/')
os.getcwd()


###Get training data, i.e. tweets about the following 4 categories: 
###"actors play", "soundtrack", "filmmaker", "movie script"

topics = ["actors play", "soundtrack", "filmmaker", "movie script"]

movie_list = topics
open_date = ["11-01", "11-01", "11-01", "11-01"]

for i in range(0, len(open_date)):
    open_date[i] = '2017-'+ open_date[i]
    
num_of_movie = len(movie_list)
for cnt in range(0,num_of_movie):
    time = 0
    while time < 1:
        try:
            if time == 0:
                cur_date = datetime.datetime.strptime(open_date[cnt],'%Y-%m-%d')
            next_date = cur_date + datetime.timedelta(days=27)
            start = cur_date.strftime('%Y-%m-%d')
            end = next_date.strftime('%Y-%m-%d')
                    
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
        except:
            print time
            pass


###Read in the four csv files we saved just now, 
#these are twitter data of the four categories.
os.chdir('/Users/duxuewei/Desktop/projectds/project/supervised_learning/')
actors_play = pd.read_csv("actors play2017-11-01.csv")
filmmaker = pd.read_csv("filmmaker2017-11-01.csv")
movie_script = pd.read_csv("movie script2017-11-01.csv")
soundtrack = pd.read_csv("soundtrack2017-11-01.csv")

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
    tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces       

    return tokens

def textToNum(theLabels,thePredLabel):
    theOutLabel = dict()
    cnt = 0
    for word in theLabels:
        theOutLabel[word] = cnt
        cnt = cnt + 1
    return str(theOutLabel[thePredLabel])

theLUT = pd.read_csv(thePathLut + 'classifierLUT.csv',index_col=0) #ALGO LUT

def optFunc(theAlgo,theParams):
    theModel = theLUT.loc[theAlgo,'optimizedCall']
    tempParam = list()
    for key, value in theParams.iteritems():
        tempParam.append(str(key) + "=" + str(value)) 
    theParams = ",".join(tempParam)
    theModel = theModel + theParams + ")"
    return theModel 

def algoArray(theAlgo):
    theAlgoOut = theLUT.loc[theAlgo,'functionCall']
    return theAlgoOut

def gridSearch(theModel):
    gridSearchOut = theLUT.loc[theModel,'gridSearch']
    return gridSearchOut

theCols = ["actors_play", "soundtrack", "filmmaker", "movie_script"]
theLabels = theCols


def find_companies(df, industry):
    return df[df.Industry == industry]



###Convert the data into the form of HW4, so that I can use the codes
###in HW 4 directly

us_companies = pd.DataFrame()
Company = pd.Series(actors_play.body.tolist() + soundtrack.body.tolist() 
    + filmmaker.body.tolist() + movie_script.body.tolist())
Industry = pd.Series(["actors_play"] * len(actors_play) 
    + ["soundtrack"] * len(soundtrack) 
    +["filmmaker"] * len(filmmaker) 
    +["movie_script"] * len(movie_script))
us_companies["Company"] = Company
us_companies["Industry"] = Industry

balanced_data = us_companies

balanced_data.reset_index(drop = True, inplace = True)


#Clean the data

finalWords = list()
theDocs = list()

for word in theCols:
    cnt = 0
    one_industry_df = find_companies(balanced_data, word)
    for company in one_industry_df.Company:
        if not isinstance(company, float):
            #print genCorpus(company)
            finalWords.append(genCorpus(company))
            theDocs.append(textToNum(theLabels,word) +"_" + str(cnt))
            cnt = cnt +  1

#Transform data into term frequency matrix
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,1))
tdm = pd.DataFrame(vectorizer.fit_transform(finalWords).toarray())

#with open('vectorizer.pk', 'wb') as fin:
#    pickle.dump(vectorizer, fin)

tdm.columns=vectorizer.get_feature_names()
tdm.index=theDocs

pca = decomposition.PCA(n_components=.95)
pca.fit(tdm)
reducedTDM = pd.DataFrame(pca.transform(tdm)) #reduced tdm distance matrix

#with open('pca.pk', 'wb') as fin:
#    pickle.dump(pca, fin)

reducedTDM.index=theDocs


pcaVar = round(sum(pca.explained_variance_ratio_),2)

fullIndex = reducedTDM.index.values
fullIndex = [int(word.split("_")[0]) for word in fullIndex]

###Train machine learning models using training set, select a best model

theModels = ["NN", "RF"]#['RF','ABDT','LOGR','NN','DT','ABDT','LDA']#,'DT','LDA','BAG','KNN','NN'] #these MUST match up with names from LUT #ABDT, #GBC, #RSM take far too long
theResults = pd.DataFrame(0,index=theModels,columns=['accuracy','confidence','runtime'])
for theModel in theModels:
    startTime = time.time()
    model = eval(algoArray(theModel))
    #model = RandomForestClassifier(random_state=50)
    print(theModel)

    #cross validation    
    cvPerf = cross_val_score(model,reducedTDM,fullIndex,cv=10)
    theResults.ix[theModel,'accuracy'] = round(cvPerf.mean(),2)
    theResults.ix[theModel,'confidence'] = round(cvPerf.std() * 2,2)
    endTime = time.time()
    theResults.ix[theModel,'runtime'] = round(endTime - startTime,0)
    
print(theResults)

#############################################
#######Run with best performing model########
#####Fine Tune Algorithm Grid Search CV######
#############################################
bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
modelChoice = theResults['accuracy'].idxmax()
#Hardcode to choose NN model becuase of running time concerns
modelChoice = "NN"
              
startTime = time.time()
model = eval(algoArray(modelChoice))
grid = GridSearchCV(estimator=model,  param_grid={"alpha": [1,0.1,0.01,0.001,0.0001,0]})#eval(gridSearch(modelChoice))
grid.fit(reducedTDM,fullIndex)
#grid.fit(train,trainIndex)
bestScore = round(grid.best_score_,4)
parameters = grid.best_params_
endTime = time.time()
print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime,0)))

############################################
######Train Best Model on Full Data Set#####
########Save Model for future use###########
############################################
startTime = time.time()
model = eval(optFunc(modelChoice,parameters)) #train fully validated and optimized model
model.fit(reducedTDM,fullIndex)
#model.fit(train,trainIndex)
joblib.dump(model, modelChoice + '.pkl') #save model
endTime = time.time()
print("Model Save Time: " + str(round(endTime - startTime,0)))

"""
os.chdir('/Users/duxuewei/Desktop/projectds/project/supervised_learning/12_1/')
os.getcwd()

corpus_13 = pd.read_csv('corpus_13.csv')
corpus_14 = pd.read_csv('corpus_14.csv')
corpus_15 = pd.read_csv('corpus_15.csv')
corpus_16 = pd.read_csv('corpus_16.csv')
corpus_17 = pd.read_csv('corpus_17.csv')
corpus_18 = pd.read_csv('corpus_18.csv')
corpus_19 = pd.read_csv('corpus_19.csv')
corpus_20 = pd.read_csv('corpus_20.csv')
corpus_21 = pd.read_csv('corpus_21.csv')
corpus_22 = pd.read_csv('corpus_22.csv')
corpus_23 = pd.read_csv('corpus_23.csv')
corpus_24 = pd.read_csv('corpus_24.csv')



all_tweets = pd.concat([corpus_13, corpus_14, corpus_15, corpus_16, corpus_17, 
                        corpus_18, corpus_19, corpus_20, corpus_21, corpus_22,
                        corpus_23, corpus_24])
    
    
"""
os.chdir('/Users/duxuewei/Desktop/projectds/project/')


###Run ranking_supervised.py after this script




