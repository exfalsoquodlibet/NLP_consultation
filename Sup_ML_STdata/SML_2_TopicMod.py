#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 22:41:23 2018

@author: alessia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 07:24:59 2018

@author: alessia
"""

### (0) Aim -------------------------------------------------------------------

# In this code, we: 
#   - compute topic modelling using LDA
#   - compute topic modelling using NMF 
#   - append to the original dataset, for each text/doc: 
#       + probability of each topic
#       + ranked topics and their ranked probabilities
#       + top n words and heir probability for each topic

# To achieve this we have create a number of user-defined functions.






### (1) Set Ups and Imports -------------------------------------------------------


# import modules

import os
import pandas as pd
from operator import itemgetter
import string
import numpy as np
import matplotlib.pyplot as plt
import re
import time


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


### Import user-created functions ---------------------------------------------

cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Code/functions')

from utils import *
from NLP_basic_functions import *
from TopicMod_functions import *

print ( dir(utils) )
print ( dir(NLP_basic_functions) )





### Set up working directory --------------------------------------------------
cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Outputs')

#pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)





########################
### Pre-processing #####
########################


### (1) Read Data -------------------------------------------------------------

text_df = pd.read_pickle('ST_preproc_2.pkl')


# Quick exploration of data
text_df.shape
text_df.dtypes
text_df.columns




### (2) Encode DV labels numerically to be "read" by ML algorithm -------------

ordered_labels = ['positive','negative']

lb_make = LabelEncoder()
text_df['rating_label'] = lb_make.fit_transform(text_df['rating'])

#check 
text_df[['rating', 'rating_label']]




### (3) Some EDA  ------------------------------------------------------------------


# Is there any NaN in text?
text_df[pd.isnull(text_df['text_nopunkt_lemmas'])]   #yep, 4 cases

text_df = text_df[pd.notnull(text_df['text_nopunkt_lemmas'])]   


# Plot most frequent words

# https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
all_words = text_df['text'].str.split(expand=True).unstack().value_counts()
all_words = all_words.to_frame().reset_index().rename(columns = {'index' : 'word', 0 : 'count'})

# get 50 more frequent words, lots of "rubbish"
all_words[:50].plot.bar(x='word')
all_words[:50]

#import seaborn as sns
#sns.countplot(y="count", data=top50w)
#from ggplot import *
#ggplot(aes(x='word', y='count'), data=top50w) + geom_bar(stats='identity')



### (4) CLEAN TEXT for LDA ----------------------------------------------------

# 1. tokenise
# 2. lower case
# 3. remove stopwords
# 4. remove non-alphabetic tokens (i.e., punctuations and numbers)
# 5. lemmatise
# . stemming?

preprocessing_pipe = combine_functions(POS_tagging
                                   ,lemmatise
                                   ,fix_neg_auxiliary
                                   ,lambda x : remove_stopwords(x, extra_stopwords = ['x', "'s", "not", 'us', 'no', 
                                                                                      'many', 'much', 'one', 'put', 'also', 'get', 'would', 'could', 'like', 'go', 'lot', 'make'])
                                   ,lambda s: [[re.sub(r'\d+','',x) for x in subs] for subs in s]
                                   ,flattenIrregularListOfLists
                                   ,remove_punctuation
                                   ,lambda x: list(filter(None, x))
                                   )   


text_df['text4lda'] = text_df['text_tokens'].apply(lambda x: preprocessing_pipe(x))

# check some texts
text_df[['text', 'text4lda']][:5]

# re-plot most frequent words

all_words = text_df['text4lda'].apply(list2string).str.split(expand=True).unstack().value_counts()
all_words = all_words.to_frame().reset_index().rename(columns = {'index' : 'word', 0 : 'count'})

# get 50 more frequent words: much better
all_words[:50].plot.bar(x='word')
all_words[:50]






####################
###### LDA #########
####################


### USING GENSIM ###


### (1) CREATE A DICTIONARY containing the number of times a word appears in the corpus of texts

from gensim import models, corpora

# i. Build a Dictionary - association word to numeric id
# assigning a unique integer id to each unique word while also collecting word counts and relevant statistics. 
dictionary = corpora.Dictionary(text_df['text4lda'])

# take a look
print(dictionary.token2id)   

# what's the vocabulary size?
len(dictionary.token2id.keys())
# 2030
  


### (2) Filter out words that occur too frequently or too rarely.

# - less than 15 texts (absolute number) or (infrquent words)
# - more than 0.5 documents (fraction of total corpus size, not absolute number).
# - after the above two steps, keep only the first 100000 most frequent tokens.
# Ref: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

max_freq = 0.5
min_wordcount = 20

dictionary.filter_extremes(no_below=min_wordcount
                           , no_above=max_freq
                           , keep_n=100000)

len(dictionary)
# 67....  too harsh filtering?



### (3) Transform the collection of texts to a numerical form: For each text, report how many many times each occurring word appears
# i.e., Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

bow_corpus = [dictionary.doc2bow(text) for text in text_df['text4lda']]
 
# Have a look at how the 1st text looks like: [(word_id, count), ...]
print( text_df[['text', 'text4lda']][1:2] )
print( bow_corpus[1] )

for i in range(len(bow_corpus[1])):
    print("Word {} (\"{}\") appears {} time.".format(bow_corpus[1][i][0],
          dictionary[bow_corpus[1][i][0]], 
          bow_corpus[1][i][1]))




### (4) FIND BEST NUMBER OF TOPICS 
    
# (i) divide corpus in training and test corpus. The test corpus will be used to calculate perplexity
    
from random import shuffle
shuffle(bow_corpus)
len(bow_corpus)

train_corpus, test_corpus = bow_corpus[:770], bow_corpus[770:]

# Number of words in the in the training set and in the test set
print(np.sum(cnt for document in train_corpus for _, cnt in document))
print(np.sum(cnt for document in test_corpus for _, cnt in document))

### Perplexity
# Ref: https://docs.google.com/viewer?a=v&pid=forums&srcid=MDEwMDM0NjcxOTk3Njc0MTA0MjMBMTQzMzY3Nzc1NTMzNDgyNjIyMzEBZnBOMFVLSG9BZ0FKATAuMwEBdjI&authuser=0
# https://groups.google.com/forum/#!topic/gensim/BDuOnCGpgOs
# http://qpleple.com/perplexity-to-evaluate-topic-models/
# Perplexity is a standard measure for estimating the performance of a probabilistic model. 
# The perplexity of a set of test words, is defined as the exponential of the negative normalized predictive likelihood under the model,

# (ii) loop on training set for several numbers of topics: 5, 10, 50, 100, 200, 400
topics_seq = list((2,3,4,5, 10, 15, 20, 30, 40, 50, 60, 70))


results_perplexity = []
for topic_n in topics_seq:
    start_time = time.time()
    # run model
    print('number of topics :  %d' % topic_n)
    
    model = models.LdaModel(corpus=train_corpus
                             , num_topics=topic_n
                             , id2word=dictionary
                             , passes = 20  # as we have a small corpus
                             , eta = 0.01 # topics are known to be word-sparse, the Dirichlet parameter of the word distributions is set small (e.g., 0.01), in which case learning is efficient.
                             , alpha = 0.1    #believed that each document is associated with few topics
                             , random_state = 1
                             )
    elapsed = time.time() - start_time
    # perplexity
    
    log_perplexity = model.log_perplexity(test_corpus)
    perplexity_test = np.exp(-log_perplexity)
    
    print('per-word likelihood bound     ', log_perplexity)
    print('perplexity : exp(-bound)                                     ', perplexity_test)
    print('elapsed time: %.3f' % elapsed)  
    print( ' ')
    results = [topic_n, perplexity_test]
    results_perplexity = np.concatenate((results_perplexity, results))
#----- end of loop
results_perplexity = np.reshape(results_perplexity, (len(topics_seq), 2))


#----- plot of perplexity versus number of topics
plt.plot(results_perplexity[:,0], results_perplexity[:,1], 'r--',)
plt.title('Perplexity')
plt.xlabel('Topics')
plt.ylabel('Perplexity')
plt.grid(True) 
plt.show()

# it seems to suggest 20 topics



# (iii) Build the lda model with he suggested number of topics

NUM_TOPICS = 20

# use default alpha and eta hyperparameters
lda_model1 = models.LdaModel(corpus=bow_corpus
                             , num_topics=NUM_TOPICS
                             , id2word=dictionary
                             , passes = 20  # as we have a small corpus
                             , eta = 0.01 # topics are known to be word-sparse, the Dirichlet parameter of the word distributions is set small (e.g., 0.01), in which case learning is efficient.
                             , alpha = 0.1    #believed that each document is associated with few topics
                             )


# (iv) Explore Topics

print("LDA Model:")
 
for idx in range(NUM_TOPICS):
    # Print the first 5 most representative words for each topic
    print("Topic #%s:" % idx, lda_model1.print_topic(idx, 5))
 
print("=" * 20)

# Really hard to understand the topics...


#### TO DO: USE TOPIC COHERENCE INSTEAD
# Topic Coherence
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_coherence_tutorial.ipynb
#  coherence measure output for the good LDA model should be more (better) than that for the bad LDA model

# ?lda_model.top_topics(corpus, num_words=20)
# Calculate the Umass topic coherence for each topic.




#######################################
#### (5) Let's choose K = 5 Topics ####
#######################################

seed = 17

NUM_TOPICS = 5

# use default alpha and eta hyperparameters
lda_model = models.LdaModel(corpus=bow_corpus
                             , num_topics=NUM_TOPICS
                             , id2word=dictionary
                             , passes = 20  # as we have a small corpus
                             , eta = 0.01 # topics are known to be word-sparse, the Dirichlet parameter of the word distributions is set small (e.g., 0.01), in which case learning is efficient.
                             , alpha = 0.1    #believed that each document is associated with few topics
                             , random_state = seed
                             )


# (i) EXPLORE TOPICS

print("LDA Model:")
 
for idx in range(NUM_TOPICS):
    # Print the first 5 most representative words for each topic
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 5))
 
print("=" * 20)
# they seem to be understandable/reasonable


# (ii) Test on a new "text" 

# let's try on a new (invented) text
text_test = ['important', 'listen', 'staff', 'concern', 'future', 'unknown']
text_test_bow = dictionary.doc2bow(text_test)

lda_model.get_document_topics(bow = text_test_bow)
list(lda_model[text_test_bow])
# the lda model seems to be doing a good job, the assigned topic makes sense



# explore parameers
#print(lda_model.print_topics(num_words=5))
#print(lda_model.show_topics())
#print(lda_model.show_topic(1))
#print(lda_model.get_topic_terms(1))


# (iii) probabilities that each text belongs to each topic --------------------

# Take a look first:
#list(lda_model[bow_corpus]) ## get probabilities that each document belongs to each topic.  some stochasticity here.

#for t_idx in range(len(bow_corpus)): 
#    print("Text #%s:" % t_idx)
#    for i in lda_model[bow_corpus[t_idx]]:
#        t_topic, t_topic_prob = i
#        print( "Topic #%s:" % t_topic, "Pr = %s" % t_topic_prob)


## Create dataframe of topic probabilities 

DTM_df = lda_dtm2df(lda_model[bow_corpus], 5)
# see my user-defined function in TopicMod_functions



# (iv) extract top n words and their probabilities for each topic, turn them into a dataframe

words_topics_dict = lda_topic_top_words(lda_mod = lda_model, n_top_words = 6)
words_topics_df = topictopwords_dict2df(words_topics_dict, orig_dataset = text_df, tech = 'lda')


# (v) Ranked topics for each doc, create a DTM dataframe 

ranked_DTM_df =  lda_ranked_topics2df(lda_mod = lda_model, corpus = bow_corpus)



# (v) join wih original dataframe -----------------------------------------------

text_df = merge_dfs(text_df, DTM_df, ranked_DTM_df, words_topics_df)



#### IMPORANT #######
#### need function/transformer to do this also for new held-out data
#### DISCUSS WITH THEODORE







################################################
################ USING NMF #####################
# Using Non-negative matrix factorization ######
################################################


# (1) CREATE A DICTIONARY and BOW CORPUS


from sklearn import decomposition

vectorizer = CountVectorizer(analyzer='word', 
                             tokenizer=word_tokenize,
                             min_df=20,         # to keep consisent with lda
                             max_df = 0.5)


dtm = vectorizer.fit_transform(text_df['text4lda'].apply(list2string)).toarray()

vocab = np.array(vectorizer.get_feature_names())

print( dtm.shape )
print( len(vocab) )



# (2) Choose number of topics and Run model

num_topics = 5
num_top_words = 6

clf = decomposition.NMF(n_components=num_topics, random_state=1)

# doc-topic matrix
doctopic = clf.fit_transform(dtm)

len(doctopic)


# (3) DTM into dataframe

# sandardise so that the topic probabilities for each doc sum to one    
doctopic = standardise_dtm_nmf(doctopic)

DTM_nmf_df = nmf_dtm2df(doctopic)




# (4) get top important words associated with topics 

# sandardised topic-word matrix so that word probabilities for each doc sum up o 1
wordtopic = standardise_twm_nmf(clf) 

word_topic_dict = nmf_topic_top_words(wordtopic, vocabulary = vocab)
word_topic_nmf_df = topictopwords_dict2df(word_topic_dict, orig_dataset = text_df, tech = 'nmf')



# (5) top ordered topics per each document  --------------------------------------

ranked_dtm_nmf_df = nmf_ranked_topics2df(doctopic, num_topics = 3)

         
          
# (6) Join NMF results to original dataframe
    
text_df = merge_dfs(text_df, DTM_nmf_df, ranked_dtm_nmf_df, word_topic_nmf_df)



##### SAVE DATA ####
text_df.to_pickle('/Users/alessia/Documents/DataScience/NLP_Project/Outputs/staff_text_sentiments_topics.pkl')

          
          
 