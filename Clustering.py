# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:35:24 2022

@author: Rahul Sharma "M.Tech (Data Science)"
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
km = KMeans(n_clusters=20)
km.fit(transformed)
labels = groups.target
plt.scatter(labels, km.labels_)
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.show()