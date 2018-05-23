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
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import bigrams
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline 

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve




# Set up working directory
cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Outputs')
pd.set_option('display.max_colwidth', -1)




### (2) Read Data -------------------------------------------------------------

text_df = pd.read_csv('semeval2017_preproc.csv')


# Quick exploration of data
text_df.shape
text_df.dtypes
text_df.columns

print(text_df.groupby('rating')\
    ['is_reply', 'count_punkt', 'count_ADJ', 'count_ADV', 
             'subjectivity_text', 'VDR_polarity_text']\
    .describe()
    )




### Encode DV labels numerically to be "read" by ML algorithm -----------------

ordered_labels = ['positive','negative']

#create labels that can be processed by the ML algorithm (i.e., Encode labels with value between 0 and (n_of_labels)-1.).


lb_make = LabelEncoder()
text_df['rating_label'] = lb_make.fit_transform(text_df['rating'])

#check original column and label column

text_df[['rating', 'rating_label']]




### Split data into Train and Test Data ---------------------------------------

#define X as text from column where the lemmatised texts are
X = text_df['text_nopunkt_lemmas'].values
#define y as text from column 'rating_label' created above
y = text_df.rating_label.values


# split data into train and test set randomly (but in a way that is repeatable with seed random_state)
# we set the test data to be 20% of the total data
# stratify the split according to labels so tha they are distributed in the train and test set
# as they are in the original dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)






### Define Pipeline -----------------------------------------------------------

# References:
# https://www.civisanalytics.com/blog/workflows-in-python-using-pipeline-and-gridsearchcv-for-more-compact-and-comprehensive-code/

# define steps in the pipeline

# (1) define parameters of count vectorizer
vec = CountVectorizer(analyzer="word",
                      stop_words='english',
                      ngram_range=(1, 2),
                      #preprocessor=None,
                      tokenizer=word_tokenize,
                      max_features=10000)

# inspect:
vec.get_stop_words()
vec.get_feature_names()[:10]  #first 10 features (unigrams and bigrams)
vec.get_params()


# (2) classifier
lr=LogisticRegression()

# inspect:
lr.get_params()


# define a scikit learn pipeline

pipe_bigram_lr_clf = Pipeline([


    ('vectorizer', vec),
    
    ('classifier', lr)     
   
])



# inspect: steps
pipe_bigram_lr_clf.named_steps





### Fit transformer/classifier Pipeline to train data (X_train and y_train) ----

# NOTE: When you call the pipeline’s fit() method, 
# it calls fit_transform() sequentially on all transformers, 
# passing the output of each call as the parameter to the next call, 
# until it reaches the final estimator, for which it just calls the fit() method

pipe_bigram_lr_clf.fit(X_train, y_train)

# inspect classes (that the transformer has learned about by fitting it to the data)
pipe_bigram_lr_clf.classes_





### Predict from unseen Test data ---------------------------------------------

# Let's predict the labels for the test data (that haven't been used to train the model)
# NOTE: Pipeline.predict() Apply transforms to the data, and predict with the final estimator
label_pred = pipe_bigram_lr_clf.predict(X_test)

# Let's see the predictions and the real values 
print('These are the predicted labels for the test data: ' + str(label_pred[:10]))
print('These are the original test labels that we tried to predict: ' + str(y_test[:10]))


# Pipeline classifier params; classifier is stored as step 2 ([1]), second item ([1])			
print('\nModel hyperparameters:\n', pipe_bigram_lr_clf.steps[1][1].get_params())

# Get predicted probabilities "to be" in each class for each sample (case)
pipe_bigram_lr_clf.predict_proba(X_test)[:10]    #first 10 cases
label_pred[:10]     #case is assigned to the class with higher probability
#y_test




### Evaluate Performance of trained model ---------------------------------------

# (1) Classification Accuracy: score the test data, aka pipeline test (mean) accuracy 
print('Test accuracy: %.3f' % pipe_bigram_lr_clf.score(X_test, y_test))


# (2) Other evaluation metrics for classification:

# Compute and print the confusion matrix and classification report
# [true-negative, false-positive, false-negative, true-positive] 
print(confusion_matrix(y_true = y_test, y_pred = label_pred))

# precision    recall  f1-score   support
print(classification_report(y_true = y_test, y_pred = label_pred))


# (3) ROC curve and AUC (the area under the ROC curve) to evaluate the model

label_pred_prob = pipe_bigram_lr_clf.predict_proba(X_test)[:,1]
print(label_pred_prob)

# AUC value
roc_auc_score(y_test, label_pred_prob)

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, label_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()









### debug classifiers and explain their predictions ----------------------------
#(only displayed in jupyter notebook)

import eli5

eli5.show_weights(lr, top=20,vec=vec,target_names=ordered_labels) 

#explain prediction of a particular text
eli5.explain_prediction(lr, X_test[1],vec=vec, top=15,target_names=ordered_labels)

#explaining the predictions of  ad hoc text. here a (kind of nonsensical) sentence based on tokens 
# more important in the 'outstanding' class
# change it to see how the classifier reacts.

eli5.explain_prediction(lr, "a person told us that having service safety actions rated as great",
                        vec=vec, top=15,target_names=ordered_labels)


