# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:42:59 2022

@author: Rahul Sharma "M.Tech (Data Science)"
"""
from sklearn.datasets import fetch_20newsgroups
groups = fetch_20newsgroups()

"""Visualization"""
import seaborn as sns
sns.distplot(groups.target)
import numpy as np
import matplotlib.pyplot as plt
plt.show()

'''Distribution plot of 500 Word Counts'''

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()
transformed = cv.fit_transform(groups.data)
print(cv.get_feature_names())
sns.distplot(np.log(transformed.toarray().sum(axis=0)))
plt.xlabel('Log Count')
plt.ylabel('Frequency')
plt.title('Distribution Plot of 500 Word Counts')
plt.show()