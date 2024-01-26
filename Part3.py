#!/usr/bin/env python
# coding: utf-8

# # Part 3: 
# 
# Students:
# - Konstantinos Nikoletos 
# - Konstantinos Plas

import pandas as pd
from tqdm import tqdm

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

train_data = pd.read_csv('./Part-3/data/train_q_3.1.csv', names=['Question'],  delimiter='\t', header=0)
test_data = pd.read_csv('./Part-3/data/test_without_labels_q_3.1.csv',  delimiter='\t')

print(train_data.shape)
print(test_data.shape)

print("Train data shape: ", train_data.shape)
print(train_data.head())

print("\nTest data shape: ", test_data.shape)
print(test_data.head())

from stop_words import get_stop_words
stop_words_pypi = set(get_stop_words('en'))

from nltk.corpus import stopwords
stop_words_nltk = set(stopwords.words('english'))

manual_stop_words = {'include', 'way', 'work', 'look', 'add', 'time', 'year', 'one', \
                     'month', 'day', 'help', 'think', 'tell', 'new', 'said', 'say',\
                     'need', 'come', 'good', 'set', 'want', 'people', 'use', 'day', 'week', 'know'}

stop_words= stop_words_nltk.union(stop_words_pypi)
stop_words = stop_words.union(manual_stop_words)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = re.sub(r'\W', ' ', str(processed_text))
    processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)
    processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)
    processed_text = re.sub(r'^b\s+', '', processed_text)

    tokens = [lemmatizer.lemmatize(word) for word in processed_text.split() if word not in stop_words]
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(tokens)

    return processed_text

# from tqdm.notebook import tqdm
import os
from tqdm.auto import tqdm  # for notebooks
tqdm.pandas()

preprocessed_file_path_train = 'pre_train_31.csv'
if not os.path.exists(preprocessed_file_path_train):
    print("[TRAIN] Preprocessing text...")
    train_data['text'] = train_data['Question'].progress_apply(preprocess_text)
    print("[TRAIN] Preprocessing text done.")
    train_data.to_csv(preprocessed_file_path_train, index=False)
else:
    print("[TRAIN] Reading from file")
    train_data = pd.read_csv(preprocessed_file_path_train)

preprocessed_file_path_test = 'pre_test_31.csv'
if not os.path.exists(preprocessed_file_path_test):
    print("[TEST] Preprocessing text...")
    test_data['text'] = test_data['Question'].progress_apply(preprocess_text)
    print("[TEST] Preprocessing text done.")
    test_data.to_csv(preprocessed_file_path_test, index=False)
else:
    print("[TEST] Reading from file")
    test_data = pd.read_csv(preprocessed_file_path_test)

# count nan values
print(test_data.isnull().sum())
test_data = test_data.dropna()


import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from datasketch import MinHashLSH, MinHash
import numpy as np


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_data['text'])
X_test_tfidf = vectorizer.transform(test_data['text'])

vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(train_data['text'])
X_test_count = vectorizer.transform(test_data['text'])


# ## Cosine Similarity: Random projection LSH family.

# ## Exact Cosine


from sklearn.metrics.pairwise import cosine_similarity


X_train = X_train_count
X_test = X_test_count

num_duplicates = 0

total_time = time.time()

Y = cosine_similarity(X_test_count, X_train_count)
num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])
# print(np.where((Y > 0.8).any(axis=1)))
print('Duplicates: {}'.format(num_duplicates))
print('Query time: {}'.format(time.time() - total_time))

num_duplicates = 0
for i, x_train in tqdm(enumerate(X_train)):
    for j, x_test in enumerate(X_test):
        if cosine_similarity(X_test[j], X_train[i]) > 0.8:
            num_duplicates +=1
print('Duplicates: {}'.format(num_duplicates))

from scipy.spatial.distance import jaccard, cosine
from sklearn import random_projection


def random_projection_hashing(train_vectors, test_vectors, k=5):

    transformer = random_projection.GaussianRandomProjection(n_components=k)

    X_train_buckets = transformer.fit_transform(train_vectors)
    X_test_buckets = transformer.transform(test_vectors)

    def transform_to_buckets(x):
        x = x > 0
        better_hash = []
        for c in x:
              better_hash.append(sum([j*(2**i) for i,j in list(enumerate(reversed(c)))]))
        return np.array(better_hash)

    train_hash_codes = transform_to_buckets(X_train_buckets)
    print(len(train_hash_codes))    
    train_data['hash'] = train_hash_codes
    buckets_train = train_data['hash'].unique()
    print("Train num of unique buckets: ", len(buckets_train))

    test_hash_codes = transform_to_buckets(X_test_buckets)
    print(len(test_hash_codes))
    test_data['hash'] = test_hash_codes
    buckets_test = test_data['hash'].unique()
    print("Test num of unique buckets: ", len(buckets_test))

    num_duplicates = 0
    num_candidates_per_test_doc = []
    for test_i, hash_value in tqdm(enumerate(test_hash_codes)):
        candidate_ids = train_data[train_data['hash']==hash_value].index
        num_candidates_per_test_doc.append(len(candidate_ids))
        
#         bucket = X_train[candidate_ids]
#         num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])
#         print(len(candidate_ids))
        for eid in candidate_ids:
            sim = cosine_similarity(X_test[test_i], X_train[eid])
            
            if sim > 0.8:
                num_duplicates+=1
#         if len(candidate_ids):
#             Y = cosine_similarity(X_test[test_i], X_train[candidate_ids])
#             num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])

    print(sum(num_candidates_per_test_doc)/len(test_hash_codes))
    
    return num_duplicates, num_candidates_per_test_doc
num_duplicates, num_candidates_per_test_doc = random_projection_hashing(X_train, X_test, 3)
print('Duplicates: {}'.format(num_duplicates))