#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:41:50 2021

@author: sristi
"""

import pandas as pd
import re
import sqlite3
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud 

# Load the complete dataset
df = pd.read_csv('emails.csv')

# Exploratory data analysis
print(df.info)
print(f"Spam count: {len(df.loc[df['spam'] == 1])}")
print(f"Spam count: {len(df.loc[df['spam'] == 0])}")
print(df.dtypes)
df.drop_duplicates(inplace=True)
print(df.shape)

# Data cleaning (using regex and not libraries)
cleaned_text = []
for text in df['text']:
    text = text.lower()
    # Remove any character other than a-z, like punctuation
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Remove lt and rt tags
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    
    # Remove digits and special chars
    text = re.sub("(\\d|\\W)+", " ", text)
    
    cleaned_text.append(text)
    
df['text'] = cleaned_text
    
    
# Removing stop words
stop_words = ['is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get', 'not', 'here', 'in', 'im', 'have', 'on', 're', 'new', 'subject']

############################################################### PART 1: WordCloud of complete dataset ########################################################################

# Prepare word cloud
wordcloud = WordCloud(width=800, height=800, background_color= "black", stopwords=stop_words, max_words=1000, min_font_size=20).generate(str(df['text']))

# Plot
fig = plt.figure(figsize=(8,8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

##############################################################################################################################################################################

############################################################### PART 2: Seperate WordClouds of spam and non-spam emails ########################################################################

# Prepare word cloud
spam_wordcloud = WordCloud(width=800, height=800, background_color= "black", stopwords=stop_words, max_words=1000, min_font_size=20).generate(str(df.loc[df['spam'] == 1]['text']))
non_spam_wordcloud = WordCloud(width=800, height=800, background_color= "white", stopwords=stop_words, max_words=1000, min_font_size=20).generate(str(df.loc[df['spam'] == 0]['text']))

# Plot
plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud)
plt.axis('off')
plt.title('Spam Wordcloud')

plt.subplot(1, 2, 2)
plt.imshow(non_spam_wordcloud)
plt.axis('off')
plt.title('Non-spam Wordcloud')
plt.show()

##############################################################################################################################################################################

############################################################### PART 3: Spam Detection ########################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score, classification_report

# Count vectorization is being used for word representation
cv = CountVectorizer()
text_repr = cv.fit_transform(df['text'])

X_train, X_test, y_train, y_test = train_test_split(text_repr, df['spam'], test_size=0.35, random_state=42)

# Training network
classifier = ensemble.GradientBoostingClassifier(max_depth=6, learning_rate=0.5)
classifier.fit(X_train, y_train)

# Predictions
predictions = classifier.predict(X_test)

# CLassification report
print(f'Classification Report: {classification_report(y_test, predictions)}')

##############################################################################################################################################################################
