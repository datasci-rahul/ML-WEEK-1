# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 10:48:48 2022

@author: Rahul Sharma "M.Tech (Data Science)"
"""
# import nltk
# nltk.download()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF
def letters_only(astr):
    return astr.isalpha()
cv = CountVectorizer(stop_words="english", max_features=500)
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
nmf = NMF(n_components=6, random_state=45).fit(transformed)
for topic_idx, topic in enumerate(nmf.components_):
    label = '{}: '.format(topic_idx)
print(label, " ".join([cv.get_feature_names_out()[i]
for i in topic.argsort()[:-9:-1]]))