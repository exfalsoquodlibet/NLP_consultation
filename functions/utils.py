#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:36:32 2018

@author: alessia
"""


import pandas as pd
import functools    




##############################################################################
### Decorators for turning the outcome of a function into a pandas.Series ####
##############################################################################

# Ref: https://gist.github.com/Zearin/2f40b7b9cfc51132851a

# option 1: it assumes the original function's ouput is not a pd.Series

def series_output(func):
    """Decorator for turning the outcome of a function into a pandas.Series """
    def wrapper(*args, **kwargs):
        return pd.Series(dict(outcome = func(*args, **kwargs)))
    return wrapper


# option 2: it does not assumes the original function's ouput is not a pd.Series

from functools import wraps

def string_to_series_out(func):
    """Decorator for turning the outcome of a function into a pandas.Series
    It first checks whether the outcome is already a pandas.Series or not"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        outcome = func(*args, **kwargs)
        if not isinstance(outcome, pd.Series):
            return pd.Series(dict(outcome = outcome))
        return outcome
    return wrapper




###############################################################################
#### Function to combine funcions into a pipeline of functions ################
###############################################################################
    

# Ref. for functolls.reduce: # https://docs.python.org/3/library/functools.html#functools.reduce

def combine_functions(*f_args):
    
    if len(f_args) == 1:          # only one function was provided
        first_func = f_args[0]
        return first_func  
   
    def combine(f, g):
        return lambda x: f(g(x))
    
    return functools.reduce(combine, f_args, lambda x: x)
    


### This one works on strings and returns a pd.Series
def combine_functions_output_series(*f_args):
    
    def combine(f, g):
        return lambda x: f(g(x))
    
    def output_series(x):
        return pd.Series(dict(outcome = x))
        
    #if len(f_args) == 1:          # only one function was provided
    #    first_func = f_args[0]
    #    list_funcs = [output_series first_func] 
    #    return functools.reduce(combine, list_funcs, lambda x:x)
    
    tuple_funcs = (output_series, ) + f_args
    return functools.reduce(combine, tuple_funcs, lambda x:x)






##########################################
#### Function to detokenise sentences ####
##########################################


def word_tokens2string_sentences(list_of_lists_of_tokens) :
    
    """
    Return a list containing a single string of text for each word-tokenised sentence.
    
    Parameters
    ----------
    - list_of_lists_of_tokens : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    """
    
    return [" ".join(sent) for sent in list_of_lists_of_tokens]



 
#####################################################################################
#### Function to transform a list of lists of strings into a list of strings ########
#####################################################################################


def list2string(list_of_strings) :
    """
    Return a string from a list of strings.
    """

    return " ".join(list_of_strings)




#####################################################
#### Function to flaten irregular lists of lists ####
#####################################################


import collections

def flattenIrregularListOfLists(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str,bytes)):
            yield from flattenIrregularListOfLists(el)
    
        else:
            yield el
 
    
    




### examples
    
if __name__ == "__main__":
    
    
    # string
    mystring = 'I do not care. I think. Maybe not'

    # pandas.DataFrame
    data = {'par_text': ['', 'I do not care. I think. Maybe not.', 'I ate too much.', 'I can see why'], 
            'id' : [111, 222, 333, 444]}
    
    df = pd.DataFrame(data)

    # Create some basic functions

    def sent_tok(string_par) :   #expects a string  
        '''Return a sentence-tokenized version of text as a list of string sentences'''
        try:
            return  string_par.split('.')      
        except:
            return []
    
    
    def word_tok(list_of_string_sents):   #expects a list of strings
        '''Return a word-tokenized version for each sentence in a list of string sentences'''
        try:
            return [elem.split() for elem in list_of_string_sents]
        except:
            return []

    
    # function pipeline example
    # create a function pipeline    
    f_pipeline = combine_functions(word_tok, sent_tok)
    
    print( word_tok(sent_tok(mystring)) )
    print( f_pipeline(mystring)  )  #yep
    
    print( df['par_text'].apply(sent_tok).apply(word_tok) )
    print( df['par_text'].apply(f_pipeline)  )   #yep  


    

    # decorator example
    @series_output
    def sent_tok_2(text) :
        try:
            return text.split('.')     #expect a string    
        except:
            return []
        
    
    print( sent_tok_2(mystring) )
    print( type(sent_tok_2(mystring)) )



    