#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
from wordcloud import WordCloud

words13 = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_13_revised.csv')
words14 = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_14_revised.csv')
words15 = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_15_revised.csv')
words16 = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_16_revised.csv')
words17 = pd.read_csv('/Users/haikundu/Desktop/COMS6998/finalproject/revised/10_17_revised.csv')

words = pd.concat([words13,words14,words15,words16,words17])

lines = words.dropna().values.tolist()
final1 = []
for text in lines:
    final1.extend(text)

final = ' '.join(final1)

final = final.replace('happy',' ')
final = final.replace('death',' ')
final = final.replace('day', ' ')


wordcloud = WordCloud(background_color="white",width=1000,height=860,margin=2).generate(final)

import matplotlib.pyplot as plt

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

