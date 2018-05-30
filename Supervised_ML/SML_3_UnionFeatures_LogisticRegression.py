#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 07:24:59 2018

@author: alessia
"""

### (0) Aim -------------------------------------------------------------------

# In this code, 
# (1) we: 
#   - compute standard bag-of-words features for the texts 
#   - extract ad-hoc features stored in Pandas DataFrame as categorical and numerical 
#     variables in separate processing pipelines.
# (2) We then combine them (with weights) using a FeatureUnion. 
# (3) Finally, we train a classifier on the combined set of features (with hyperparameter tuning).

# To achieve (1), we create a number of user-defined Transformers.






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
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer


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

#check 
text_df[['rating', 'rating_label']]




### (4) Split data into Training Set (training and validation) and ------------
### Test/Hold-Out Set (for evaluation)

# keep only relevant columns
df = text_df[['unique_id', 'rating_label', 'rating', 'is_reply','count_punkt','count_ADJ', 'count_ADV', 'subjectivity_text','VDR_polarity_text','text_nopunkt_lemmas']]

df_train, df_test = train_test_split(df, random_state=11)

X_train = df_train[['unique_id', 'rating', 'is_reply','count_punkt','count_ADJ', 'count_ADV', 'subjectivity_text','VDR_polarity_text','text_nopunkt_lemmas']]
y_train = df_train.rating_label.values

X_test = df_test[['unique_id', 'rating', 'is_reply','count_punkt','count_ADJ', 'count_ADV', 'subjectivity_text','VDR_polarity_text','text_nopunkt_lemmas']]
y_test = df_test.rating_label.values






### (5) Adding AD-HOC FEATURES ------------------------------------------------

# References: 
# https://www.slideshare.net/PyData/julie-michelman-pandas-pipelines-and-custom-transformers
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

# Custom Transformers

# NOTE: BaseEstimator is included to inherit get_params() which is needed for Grid Search

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step. 
    This class selects a column from a pandas data frame.
    """
    
    # initialise
    def __init__(self, columns):
        self.columns = columns          # e.g. pass in a column name to extract

    def fit(self, df, y=None): 
        return self                     # does nothing

    def transform(self, df):            #df: dataset to pass to the transformer
        
        df_cols = df[self.columns]    #equivelent to df['columns']
        return df_cols

    
    
class CatToDictTransformer(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step. 
    This class turns columns from a pandas data frame (type Series) that
    contain caegorical variables into a list of dictionaries 
    that can be inputted into DictVectorizer().
    """
    
    # initialise
    def __init__(self):
        self              #could also use 'pass'
        
    # 
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        #X: dataset to pass to the transformer.
        Xcols_df = pd.DataFrame(X)
        Xcols_dict = Xcols_df.to_dict(orient = 'records')
        return Xcols_dict


import itertools

class Series2ListOfStrings(BaseEstimator, TransformerMixin):
    
    """
    Class for building sklearn Pipeline step. 
    This class turns columns from a pandas data frame (type Series) that
    contain lists of string sentences into a list of strings 
    that can be inputted into CountVectorizer().
    """
    # initialise
    def __init__(self):
        self              
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        #X: dataframe to pass to the transformer.
        Xvals = X.values
        Xstrs = list(itertools.chain.from_iterable(Xvals))   #flatten nested list
        return Xstrs


    

# TODO: create an alternative Transformer to dummy code caegorical variables
        # using One Hot Encoder (need to integer encode first)
        # using get_dummy (so allowing for k-1 levels)




    
### (6) Initiate Pipelines ----------------------------------------------------   

# pipeline to extract ad-hoc features (saved as dataframe columns) from data
pipe_adhoc_features = Pipeline([
        
        ('adhoc', FeatureUnion([
                
                # pipeline for categorical features
                ('cat', Pipeline([
                        ('selector', ColumnSelector(columns=['is_reply'])),
                        ('todictionary', CatToDictTransformer()),
                        ('dv', DictVectorizer())
                        ])),
    
                # pipeline for numerical features
                ('num', Pipeline([
                        ('selector', ColumnSelector(columns=['count_punkt', 'count_ADJ', 'count_ADV', 'subjectivity_text', 'VDR_polarity_text'])),
                        ('scaler', StandardScaler())
                        ]))
        ]))
    ])


    
# Pipeline for text-features (bag-of-words)

vec = CountVectorizer(analyzer="word",
                      ngram_range = (1,3),
                      stop_words='english',
                      tokenizer=word_tokenize,
                      max_features=10000
                      )

pipe_bags_words = Pipeline([
        
        ('selector', ColumnSelector(columns=['text_nopunkt_lemmas'])),
        
        ('transformer', Series2ListOfStrings()),
        
        ('vectorizer', vec),
        
        ('tf_idf', TfidfTransformer())
        
        ])
  


### Instantiate final Pipeline
    
    
lr = LogisticRegression()


pipe_lr_clf = Pipeline([
        
        # Combined text (bag-of-word) and ad-hoc features
        ('features', FeatureUnion(
                
                transformer_list = [
                        
                        ('adhoc', pipe_adhoc_features),
                
                        ('text_bow', pipe_bags_words)
                        
                        ],
                
                # weight components in FeatureUnion
                transformer_weights={
                        'ad_hoc': 0.6,
                        'text_bow': 1.0
                        }
                        
                )),
        
        # Use classifier on combined features
        ('classifier', lr)
        
        ])



  


### (7) Hyperparameters tuning ------------------------------------------------


# Let's perform grid search on classifier in pipeline

# Define parameters space and dictionary

parameters = dict(
        
        #vectorizer__ngram_range = [(1,2), (1,3)],
        
        #vectorizer__max_features = np.arange(5000, 11000, step=1000),
        
        classifier__C = [0.01, 0.1, 1, 10, 100],    # too ambitious: np.logspace(-5, 8, 15), 
        
        classifier__penalty = ['l1', 'l2']
        
        )


# Instantiate the GridSearchCV object: cv

pipe_lr_clf_cv = GridSearchCV(estimator=pipe_lr_clf,
                              param_grid=parameters,
                              cv=3,
                              scoring='accuracy'    #could be smth else, e.g., "neg_log_loss"
                              )


# Fit it to the data 
pipe_lr_clf_cv.fit(X_train, y_train)






### (8) Prediction on test data -----------------------------------------------

print("Tuned pipeline's Hyperparameters: {}".format(pipe_lr_clf_cv.best_params_))
print("Training accuracy or best score: {}".format(pipe_lr_clf_cv.best_score_))

pipe_lr_clf_cv.best_estimator_


# inspect:
# pipe_lr_clf_cv.get_params()
# pipe_lr_clf_cv.estimator

       

# Inspect results for each combination of parameters' value

cv_results = pipe_lr_clf_cv.cv_results_
cv_results.keys()

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(params, mean_score)





### (9) Evaluation ------------------------------------------------------------

# Let's predict from the "best model"
pipe_lr_clf_cv.score(X_test, y_test)    #testing accuracy

y_predictions = pipe_lr_clf_cv.predict(X_test)

print(classification_report(y_true = y_test, y_pred = y_predictions ))

# [true-negative, false-positive, false-negative, true-positive] 
print(confusion_matrix(y_true = y_test, y_pred = y_predictions))

#ROC curve and AUC (the area under the ROC curve) to evaluate the model
y_pred_prob = pipe_lr_clf_cv.predict_proba(X_test)[:,1]
print(y_pred_prob)

# AUC value
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














