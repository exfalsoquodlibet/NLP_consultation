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
import numpy as np


pd.set_option('display.max_colwidth', -1)


# set up working directory 

cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Data')




### Read Data -------------------------------------------------------------

apr = pd.read_csv('Apr17stafftalks.csv', header=0)
jul = pd.read_csv('July17stafftalks.csv', header=0)
other = pd.read_csv('ST.csv', header=0)

apr.rename(columns = {'Postive':'Positive'}, inplace = True)
jul.rename(columns = {'Postive':'Positive'}, inplace = True)
other.rename(columns = {'Text':'text'}, inplace = True)

apr.sort_values('text', inplace = True)
jul.sort_values('text', inplace = True)
other.sort_values('text', inplace = True)


talks_df = pd.concat([apr, jul, other])

talks_df.shape

talks_df.sort_values('text', inplace = True)

# there are many null-values in text
talks_df = talks_df[pd.notnull(talks_df['text'])]
# and also null values included as "-"
talks_df[talks_df.text == "-"].shape  #220

talks_df = talks_df[talks_df.text != "-"]

talks_df.shape




### De-duplicate data --------------------------------------------------

# number of unique ratings or raters(?)
talks_df.text.value_counts()    #1021 unique texts, so there are duplicate texts
talks_df.text.isnull().sum()    #0, ok

talks_df = talks_df.drop_duplicates(subset = 'text', keep='first')



# add a unique_id variable
talks_df['unique_id'] = np.arange(1, talks_df.shape[0]+1)
    


### Classify texts ------------------------------------------------------------

talks_df['rating'] = np.where(talks_df.Positive > talks_df.Negative, 'pos', np.where(talks_df.Positive < talks_df.Negative, 'neg', 'neutral'))


talks_df.groupby('rating').describe()

#          count      
#rating                                      ...                                  
#neg      444.0    
#neutral  151.0    
#pos      426.0 




### Remove neutral ratings ---------------------------------------------------------

talks_df = talks_df[talks_df.rating != 'neutral']




#### SAVE DATA ----------------------------------------------------------------

talks_df.to_csv('/Users/alessia/Documents/DataScience/NLP_Project/Outputs/ST_preproc.csv')






