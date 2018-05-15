#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 06:00:57 2018

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
from sklearn.feature_extraction.text import HashingVectorizer


# Set up working directory
cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Data')
pd.set_option('display.max_colwidth', -1)





### (1) Import user-defined basic NLP functions -------------------------------

# add directory with functions to sys.path  (DOES NOT WORK)
# https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath 

#import sys
#sys.path.insert(0, "Users/alessia/Documents/DataScience/textconsultations")

#print(sys.path)
#dir(sys)

cwd = os.chdir('/Users/alessia/Documents/DataScience/textconsultations')

import nlpfunctions.basic_NLP_functions as b_nlp

dir(b_nlp)

cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Data')




### (2) Read Data -------------------------------------------------------------

text_df = pd.read_table('semeval2017.txt', header=None,  
                          names=('rater_id', 'rating', 'text', 'unknown'))


# Quick exploration of data

text_df.shape
# n = 20632

# number of unique ratings or raters(?)
text_df.rater_id.value_counts()    #20556 unique id, so probably raters
text_df.rater_id.isnull().sum()

# add a unique_id variable
text_df['unique_id'] = np.arange(1, text_df.shape[0]+1)
    
text_df.rating.value_counts()
# neutral     10342
# positive    7059 
# negative    3231

text_df.groupby('rating').describe()




### (3) Data Cleaning & Pre-Processing ---------------------------------------------------------


# (i) Remove neutral ratings

text_df = text_df[text_df.rating != 'neutral']

text_df.text[:10]


# (ii) Transform all text to lower characters

text_df['text2'] = text_df['text'].str.lower()


# (iii) If text contains @<whatever> then mark it as 'reply' (can be useful info for classification)

text_df['is_reply'] = text_df.text2.apply(lambda x: any(pd.Series(x).str.contains('@\S+')))


# (iv) # Remove (alternaively, count how many there are in a text before removing:
    # URL (http and www)
    # hashtags (#)
    # @

text_df['text2'] = text_df['text2'].str.replace('http\S+|www.\S+|@\S+|#\S+', '', case=False).str.lstrip()


# (iv) Sentence- and Word-Tokenise 
       
text_df['text2_tok_sent'] = text_df['text2'].apply(lambda x: b_nlp.sent_tokenise_df(x))        

text_df['text2_tok_word'] = text_df['text2_tok_sent'].apply(lambda x: b_nlp.word_tokenise_df(x)) 



# (v) Count meaningful punctuation in each text        
        
punkt_list = '!?:-)('

text_df['count_punkt'] = pd.Series([len([word for word in str(sent) if word in punkt_list]) for 
       sent in text_df['text2']], index=text_df.index)


# (vi) Count number of adjectives and adverbs

# pos_tag -> count

text_df['text2_pos_tag'] = text_df['text2_tok_word'].apply(lambda x: b_nlp.POS_tagging_df(x)) 


text_df['count_ADJ'] = pd.Series([sum([len([token for token, pos in entry if pos.startswith('J')]) for 
       entry in sent]) for sent in text_df['text2_pos_tag']], index=text_df.index)

text_df['count_ADV'] = pd.Series([sum([len([token for token, pos in entry if pos.startswith('R')]) for 
       entry in sent]) for sent in text_df['text2_pos_tag']], index=text_df.index)



# (vii) subjective score

text_df['subjectivity_sent'] = text_df['text2_tok_sent'].apply(lambda x: b_nlp.get_subjectivity_df(x)) 

text_df['subjectivity_text'] = text_df['subjectivity_sent'].apply(lambda x: np.nanmean(x))



# (viii) Vader polariy score

text_df['VDR_polarity_sent'] = text_df['text2_tok_sent'].apply(lambda x: b_nlp.get_sentiment_score_df(x)) 

text_df['VDR_polarity_text'] = text_df['VDR_polarity_sent'].apply(lambda x: np.nanmean(x))



# (ix) Remove unnecessary punctuation -> POS_tag again -> lemmatise     
   
text_df['text2_nopunkt_tok_sent'] = text_df['text2_tok_sent'].apply(lambda x: b_nlp.remove_punctuation_df(x, item_to_keep=""))

text_df['text2_nopunkt_tok_word'] = text_df['text2_tok_sent_nopunkt'].apply(lambda x: b_nlp.word_tokenise_df(x))
        
text_df['text2_nopunkt_pos_tag'] = text_df['text2_nopunkt_tok_word'].apply(lambda x: b_nlp.POS_tagging_df(x))         

# remove stopwords from tuples?

#pd.Series([[[token for token, pos in entry if token not in stopwords.words('english')] for 
#       entry in sent] for sent in text_df['text2_nopunkt_pos_tag']], index=text_df.index)

text_df['text2_nopunkt_lemmas'] = text_df['text2_nopunkt_pos_tag'].apply(lambda x: b_nlp.lemmatise_df(x))         


# (x) flatten lists of string sentences

text_df['text_nopunkt_lemmas_b'] = text_df['text2_nopunkt_lemmas'].apply(lambda x: b_nlp.word_detokenise_sent_df(x))

text_df['text_nopunkt_lemmas_c'] = text_df['text_nopunkt_lemmas_b'].apply(lambda x: b_nlp.list2string_df(x))

text_df['text_nopunkt_lemmas'] = pd.Series(["".join(x) for x in text_df['text_nopunkt_lemmas_c']], index=text_df.index)



# remove digits or just replace with "digit"
# what to do when people do not separate words (e.g., zionist, zionistthat, zionisttheir)?



### We should have everything in place now for some supervised ML fun!



### (4) Vectorisation & Document-Term-Matrix ----------------------------------

count_vect = CountVectorizer()
# count_vect

X_transformed = count_vect.fit_transform(text_df.text_nopunkt_lemmas.values)
# sparse array

# by using .A we show the sparse matrix as an numpy array
X_transformed.A

# we transpose the matrix, so that each unique text in a column
# and features (tokens) are rows
# we append a new column with the feature names with .get_feature_names() to make it even more clear.

# DTM *document term matrix*

vectdf =pd.DataFrame(X_transformed.A.T)
vectdf['term']=count_vect.get_feature_names()

#Show the words and the its numerical index
count_vect.vocabulary_

# “Term Frequency times Inverse Document Frequency”
tfidf_vect = TfidfVectorizer()
X_tf_transformed = tfidf_vect.fit_transform(text_df.text_nopunkt_lemmas.values)

tfidf_vectdf =pd.DataFrame(X_tf_transformed.A.T)
tfidf_vectdf['term']=tfidf_vect.get_feature_names()
tfidf_vectdf


