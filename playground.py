#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:57:11 2018

@author: alessia
"""

import os
import pandas as pd

os.chdir("Documents/DataScience/NLP_Project/Data")

os.listdir()



#########

# References
# https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75

##########


rater1 = pd.read_csv('~/Downloads/AT_sa_q1_sample.csv')

rater2 = pd.read_csv('~/Downloads/TM_sa_q1_sample copy.csv')

rater1.columns = ['index', 'Respondent_ID', 'Q1_census_methods', 'AT']
rater2.columns = ['index', 'Respondent_ID', 'Q1_census_methods', 'Unnamed',
       'TM']

rater2.TM = rater2.TM.apply(lambda x : rescale_to_01_df(x, -1, 1))

rater2.TM = rater2.TM.apply(lambda x : 1 if x > 0.5 else 0)


raters = pd.merge(rater2.loc[:,['index', 'Respondent_ID', 'Q1_census_methods', 'TM']], rater1.loc[:,['index', 'Respondent_ID', 'Q1_census_methods', 'AT']], 
                  how='inner', on=['index', 'Respondent_ID', 'Q1_census_methods'])

raters

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(raters.AT, raters.TM)    #.26 (slight/fair agreement)

# => A reassess- ment was performed and the kappa statistic after the “learn and improve” process was (substantial agreement).





