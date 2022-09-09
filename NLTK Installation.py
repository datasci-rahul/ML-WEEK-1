# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 10:48:48 2022

@author: Rahul Sharma "M.Tech (Data Science)"
"""

'''Downloading NLTK'''
# import nltk
# nltk.download()


'''importing names'''

from nltk.corpus import names
print(names.words()[:10])

'''length of words'''

print(len(names.words()))

'''     PorterStemmer      '''

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
words=['machines', 'learning','stating', 'owned', 'agreed','reference']
for w in words:
    print(w, " : ", ps.stem(w))
    
    
'''    WordNetLemmatizer    '''
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print("machines :", wnl.lemmatize("machines"))
print("learning :", wnl.lemmatize("learning"))