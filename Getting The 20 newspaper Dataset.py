# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 08:39:45 2022

@author: Rahul Sharma "M.Tech (Data Science)"
"""

'''fetching 20 newsgroups dataset'''

from sklearn.datasets import fetch_20newsgroups
groups = fetch_20newsgroups()


'''thinking about feature'''

print(groups.keys())
print(groups['target_names'])
print(groups.target)

import numpy as np
print(np.unique(groups.target))
print(groups.data[0])
print(groups.target[0])
print(groups.target_names[groups.target[0]])
print(len(groups.data[0]))
print(len(groups.data[1]))