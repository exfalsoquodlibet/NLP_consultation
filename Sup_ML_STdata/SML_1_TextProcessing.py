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


pd.set_option('display.max_colwidth', -1)




### (1) Import user-defined basic NLP functions and utils -------------------------------



cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Code/functions')

import utils
import NLP_basic_functions

print ( dir(utils) )
print ( dir(NLP_basic_functions) )





# Set up working directory
cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Outputs')




### (2) Read Data -------------------------------------------------------------

data_file = '/Users/alessia/Documents/DataScience/NLP_Project/Outputs/ST_preproc.csv'


text_df = pd.read_csv(data_file, header=0)

text_df.shape

text_df.groupby('rating').describe()

#          count      
#rating                                      ...                                  
#neg      444.0    
#pos      426.0 



### Create text-processing pipelines ------------------------------------------



# (iv) Sentence- and Word-Tokenise 
    
text_pipe_1 = combine_functions(sent_tokenise, word_tokenise, to_lower)

text_df['text_tokens'] = text_df['text'].apply(lambda x: text_pipe_1(x))

text_df['text_sents'] = text_df['text_tokens'].apply(lambda x: word_tokens2string_sentences(x))       


# (v) Count meaningful punctuation in each text        
        
punkt_list = ["!", "?", "..."]

text_df['count_punkt'] = text_df['text_tokens'].apply(lambda x: count_punkt(x, punkt_list = punkt_list))



# (vi) Count proportion of adjectives and adverbs

# pos_tag -> count

count_pos_pipe = combine_functions(POS_tagging, count_pos)

text_df['prop_ADJ'] = text_df['text_tokens'].apply(lambda x: count_pos_pipe(x, pos_to_cnt = 'J')) 

text_df['prop_ADV'] = text_df['text_tokens'].apply(lambda x: count_pos_pipe(x, pos_to_cnt = 'R')) 






# (vii) subjective score
subj_pipe = combine_functions(get_subjectivity, np.nanmean)

text_df['subjectivity_text'] = text_df['text_sents'].apply(lambda x: subj_pipe(x)) 



# (viii) Vader polariy score

vdr_pipe = combine_functions(get_sentiment_score_VDR, np.nanmean)

text_df['VDR_polarity_text'] = text_df['text_sents'].apply(lambda x: vdr_pipe(x)) 




# (ix) Remove unnecessary punctuation -> POS_tag again -> lemmatise  

lemmatise_pipe = combine_functions(word_tokenise, 
                                   POS_tagging,
                                   lemmatise,
                                   fix_neg_auxiliary,
                                   mark_negation,
                                   lambda x : remove_stopwords(x, extra_stopwords = ['x', "'s"]),
                                   flattenIrregularListOfLists,
                                   remove_punctuation,
                                   list2string,
                                   lambda x : " ".join(x.split()) 
                                   )   
 
# this only works if remove_stopwords is last one in the pipeline  
#text_df['text_nopunkt_lemmas'] = text_df['text_sents'].apply(lambda x: lemmatise_pipe(x, extra_stopwords = ['x', "'s"]))

text_df['text_nopunkt_lemmas'] = text_df['text_sents'].apply(lambda x: lemmatise_pipe(x))






### We should have everything in place now for some supervised ML fun!

text_df.to_pickle('/Users/alessia/Documents/DataScience/NLP_Project/Outputs/ST_preproc_2.pkl')





