#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:00:23 2018

@author: alessia
"""

### (0) Set Ups and Imports -------------------------------------------------------


# import modules

import os
import pandas as pd
from operator import itemgetter
import string
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import bigrams
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline 

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score


# Set up working directory
cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Outputs')
pd.set_option('display.max_colwidth', -1)




### (2) Read Data -------------------------------------------------------------

text_df = pd.read_csv('semeval2017_preproc.csv')


# Quick exploration of data
text_df.shape
text_df.dtypes
text_df.columns


### 

ordered_labels = ['positive','negative']

#create labels that can be processed by the ML algorithm (i.e., Encode labels with value between 0 and (n_of_labels)-1.).


lb_make = LabelEncoder()
text_df['rating_label'] = lb_make.fit_transform(text_df['rating'])

#check original column and label column

text_df[['rating', 'rating_label']]





### ---------------------------------------------------------------------------

#define parameters of count vectorizer
vec = CountVectorizer(analyzer="word",stop_words='english',
                                   ngram_range=(1, 2),
                                   tokenizer=word_tokenize,
                                   max_features=10000)


lr=LogisticRegression()


#define a scikit learn pipeline

bigram_clf = Pipeline([


    ('vectorizer', vec),
    
    ('classifier', lr)     # or any other classifier deemed useful!
   
])





### fit 

###

#define X as text from column where the lemmatised texts are
X = text_df['text_nopunkt_lemmas'].values
#define y as text from column 'rating_label' created above
y = text_df.rating_label.values


###

# split data into train and test set randomly (but in a way that is repeatable with seed random_state)
# we set the test data to be 20% of the total data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit the train data into our transformer/classifier pipeline

print(bigram_clf.fit(X_train, y_train))




# Let's predict the labels for the test data (that haven't been used to train the model)
label_pred = bigram_clf.predict(X_test)

# Let's see the predictions and the real values 
print('These are the predicted labels for the test data: ' + str(label_pred))
print('These are the original test labels that we tried to predict: ' + str(y_test))

# The model did ok: it predicted 3 out of 4 labels correctly

#score the test data

print(bigram_clf.score(X_test, y_test))

# let's calculate the AUC (the area under the ROC curve) to evaluate the model

label_pred_prob = bigram_clf.predict_proba(X_test)[:,1]
print(label_pred_prob)

roc_auc_score(y_test, label_pred_prob)




