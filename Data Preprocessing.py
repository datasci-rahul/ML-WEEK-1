# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:59:36 2022

@author: Rahul Sharma "M.Tech (Data Science)"
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
def letters_only(astr):
    return astr.isalpha()

cv = CountVectorizer(stop_words="english", max_features=50)
groups = fetch_20newsgroups()
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()
for post in groups.data:
    cleaned.append(' '.join([
lemmatizer.lemmatize(word.lower())
                             
for word in post.split()
if letters_only(word) 
and word not in all_names]))
transformed = cv.fit_transform(cleaned)
print(cv.get_feature_names_out())