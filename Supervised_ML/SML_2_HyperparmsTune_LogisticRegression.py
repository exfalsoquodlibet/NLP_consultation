#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:08:33 2018

@author: alessia
"""

### (1) Set Ups and Imports -------------------------------------------------------


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

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline 

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

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




### (3) Encode DV labels numerically to be "read" by ML algorithm -----------------

ordered_labels = ['positive','negative']

lb_make = LabelEncoder()
text_df['rating_label'] = lb_make.fit_transform(text_df['rating'])

#check original column and label column

text_df[['rating', 'rating_label']]




### (4) Split data into Training Set (training and validation) and Test/Hold-Out Set (for evaluation)

# Stratify the split according to labels to mirror original distribution in both training and test data

#define X as text from column where the lemmatised texts are
X = text_df['text_nopunkt_lemmas'].values
#define y as text from column 'rating_label' created above
y = text_df.rating_label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=11)



### (5) Hyperparameters Tuning wih GridSearchCV

# References:
# http://www.cse.chalmers.se/~richajo/dit865/files/Automatic%20hyperparameter%20tuning.html
# https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/fine-tuning-your-model?ex=10
# https://gist.github.com/mneedham/3936d657b50b7c07cd3fe0c8d8c71496
# # https://www.civisanalytics.com/blog/workflows-in-python-using-pipeline-and-gridsearchcv-for-more-compact-and-comprehensive-code/


# Which hyperparameer to tune?

# Logistic regression has a regularization parameter: C 
# C controls the inverse of the regularization strength, 
# A large C can lead to an overfit model, while a small C can lead to an underfit model.
# We can also tune the type of penality parameter: [l1, l2]]

# CountVectorizer (first transformer)
# We could vary number of n-grams
# and try to remove and non-remove stop-word
# or number of max_features

# To avoid local optima, I should try all the combinations of parameters, and not just vary them independently.

# GridSearchCV allows to construct a grid of all the combinations of parameters, 
# tries each combination, and then reports back the best combination/model


### (5.1) On the Pipeline

# Define Pipeline

vec = CountVectorizer(analyzer="word",
                      stop_words='english',
                      tokenizer=word_tokenize
                      )

# vec.get_params().keys()

lr = LogisticRegression()


# Instantiate pipeline

pipe_lr_clf = Pipeline([
        
        ('vectorizer', vec),
        
        ('classifier', lr)
        
        ])
        

# Define parameters space and dictionary

parameters = dict(
        
        vectorizer__ngram_range = [(1,2), (1,3)],
        
        vectorizer__max_features = np.arange(5000, 11000, step=1000),
        
        classifier__C = [0.01, 0.1, 1, 10, 100],    # too ambitious: np.logspace(-5, 8, 15), 
        
        classifier__penalty = ['l1', 'l2']
        
        )


# Instantiate the GridSearchCV object: cv

pipe_lr_clf_cv = GridSearchCV(estimator=pipe_lr_clf,
                              param_grid=parameters,
                              cv=3,
                              scoring='accuracy'    #could be smth else, e.g., "neg_log_loss"
                              )
 

# inspect:
# pipe_lr_clf_cv.get_params()
# pipe_lr_clf_cv.estimator

       

### Fit it to the data --------------------------------------------------------
pipe_lr_clf_cv.fit(X_train, y_train)



# Predict on test data and evaluate

print("Tuned pipeline's Hyperparameters: {}".format(pipe_lr_clf_cv.best_params_))
print("Training accuracy or best score: {}".format(pipe_lr_clf_cv.best_score_))

pipe_lr_clf_cv.best_estimator_



# Inspect results for each combination of parameters' value

cv_results = pipe_lr_clf_cv.cv_results_
cv_results.keys()

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(params, mean_score)



### Evaluation ---------------

vec = CountVectorizer(analyzer="word",
                      stop_words='english',
                      ngram_range = (1, 3),
                      max_features = 9000,
                      tokenizer=word_tokenize
                      )

# vec.get_params().keys()

lr = LogisticRegression(C =1, penalty='l1')


# Instantiate pipeline

best_pipe_lr_clf = Pipeline([
        
        ('vectorizer', vec),
        
        ('classifier', lr)
        
        ])



# Let's predict from the "best model"
pipe_lr_clf_cv.score(X_test, y_test)    #testing accuracy

y_predictions = pipe_lr_clf_cv.predict(X_test)

print(classification_report(y_true = y_test, y_pred = y_predictions ))

# [true-negative, false-positive, false-negative, true-positive] 
print(confusion_matrix(y_true = y_test, y_pred = y_predictions))

#ROC curve and AUC (the area under the ROC curve) to evaluate the model
y_pred_prob = pipe_lr_clf_cv.predict_proba(X_test)[:,1]
print(y_pred_prob)

# AUC value (best)
print(roc_auc_score(y_test, y_pred_prob))

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# calculate log_loss based on prediction probability and true value 
# (check TM's code)








### (5.2) JUST AS AN EXAMPLE on a single object, the classifier ('estimator')

# Define hyperparameter space for lr()
c_space = np.logspace(-5, 8, 15)    #could equally set to [0.01, 0.1, 1, 10, 100]
p_space = ['l1', 'l2']

# Setup the hyperparameter grid
param_grid = {'C': c_space, 'penalty': p_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# inspect:
# logreg_cv.get_params()
# logreg_cv.estimator

X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(text_df.subjectivity_text.values.reshape(-1,1), 
                                                                y, test_size=0.20, stratify=y, random_state=19)


# Fit it to the data (let's predict labels based on subjectivity scores)
logreg_cv.fit(X_train_ex, y_train_ex)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
# training accuracy
print("Best score is {}".format(logreg_cv.best_score_))   

# Let's predict from the "best model"
y_predictions = logreg_cv.predict(X_test_ex)
print(classification_report(y_true = y_test_ex, y_pred = y_predictions ))
logreg_cv.score(X_test_ex, y_test_ex)    #testing accuracy






