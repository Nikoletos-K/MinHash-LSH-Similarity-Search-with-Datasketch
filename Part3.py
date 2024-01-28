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

import os
from tqdm.auto import tqdm  # for notebooks
tqdm.pandas()

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from datasketch import MinHashLSH, MinHash
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard, cosine
from sklearn import random_projection

def exact_cosine(train_vectors, test_vectors):
    total_time = time.time()
    num_duplicates = 0
    Y = cosine_similarity(test_vectors, train_vectors)
    num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])
    print('Duplicates: {}'.format(num_duplicates))
    print('Query time: {}'.format(time.time() - total_time))
    print("Cardinality: ", train_vectors.shape[0]*test_vectors.shape[0])

    return num_duplicates

def random_projection_hashing(train_vectors, test_vectors, train_data, k=5):

    total_time = time.time()
    build_time = time.time()
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
    train_data['hash'] = train_hash_codes
    buckets_train = train_data['hash'].unique()

    test_hash_codes = transform_to_buckets(X_test_buckets)
    
    build_time = time.time() - build_time    
    query_time = time.time()
    num_duplicates = 0
    num_candidates_per_test_doc = []
    for test_i, hash_value in tqdm(enumerate(test_hash_codes)):
        candidate_ids = train_data[train_data['hash']==hash_value].index
        num_candidates_per_test_doc.append(len(candidate_ids))
        if len(candidate_ids):
            Y = cosine_similarity(test_vectors[test_i], train_vectors[candidate_ids])
            num_duplicates += len(np.where((Y > 0.8).any(axis=1))[0])
    print('Duplicates: {}'.format(num_duplicates))
    print('Build time: {}'.format(build_time))
    print('Query time: {}'.format(time.time() - query_time))
    print('Total time: {}'.format(time.time() - total_time))
    print("Cardinality: ", train_vectors.shape[0]*test_vectors.shape[0])
    
    return num_duplicates, num_candidates_per_test_doc

def exact_jaccard_vectors(train_vectors, test_vectors):
    total_time = time.time()    
    def jaccard_similarity(x,y):
        return 1 - jaccard(x,y)
    
    num_duplicates = 0    
    for i in tqdm(range(test_vectors.shape[0])):
        for j in range(train_vectors.shape[0]):
            similarity = jaccard_similarity(test_vectors[i], train_vectors[j])
            if similarity > 0.8:
                num_duplicates += 1
                break

    print('Duplicates: {}'.format(num_duplicates))
    print('Query time: {}'.format(time.time() - total_time))
    print("Cardinality: ", train_vectors.shape[0]*test_vectors.shape[0])

    return num_duplicates

    
def jaccard_similarity(x: str, y: str):
    x_set = set(x.lower().split())
    y_set = set(y.lower().split())
    intersection = len(x_set.intersection(y_set))
    union = len(x_set.union(y_set))
    similarity = intersection / union

    return similarity

def exact_jaccard_tokens(train_data, test_data):
    total_time = time.time()
    
    num_duplicates = 0
    for i, test_doc in tqdm(enumerate(test_data)):
        for j, train_doc in enumerate(train_data):
            similarity = jaccard_similarity(test_doc, train_doc)
            if similarity > 0.8:
                num_duplicates += 1
                break
    
    print('Duplicates: {}'.format(num_duplicates))
    print('Query time: {}'.format(time.time() - total_time))
    print("Cardinality: ", len(train_data)*len(test_data))

    return num_duplicates

def lsh_minhash(train_data, test_data, num_permutations=[16, 32, 64], threshold=0.8):
        
    duplicates_per_num_perm = []
    cardinalities = []

    for num_perm in num_permutations:
        num_duplicates = 0
        print(f"\n\nCalculating LSH for num_perm={num_perm} and threshold={threshold}...")
        total_time = time.time()
        build_time = time.time()
        
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        minhash_signatures_train = []
        bucket_sizes = []
        for i, doc in tqdm(enumerate(train_data)):
            minhash = MinHash(num_perm=num_perm)
            for word in set(doc.split()):
                minhash.update(word.encode('utf8'))
            minhash_signatures_train.append(minhash)

        for i, minhash in enumerate(minhash_signatures_train):
            lsh.insert(i, minhash)
        build_time = time.time() - build_time
        
        query_time = time.time()
        for i, doc in tqdm(enumerate(test_data), desc="Quering for each doc file"):
            minhash = MinHash(num_perm=num_perm)
            for word in set(doc.split()):
                minhash.update(word.encode('utf8'))

            candidates = lsh.query(minhash)
            bucket_sizes.append(len(candidates))            
            for candidate in candidates:
                similarity = jaccard_similarity(doc, train_data[candidate])
                if similarity > 0.8:
                    num_duplicates += 1
                    break
        
        duplicates_per_num_perm.append(num_duplicates)
        cardinalities.append(sum(bucket_sizes))
        print('Duplicates: {}'.format(num_duplicates))
        print('Build time: {}'.format(build_time))
        print('Query time: {}'.format(time.time() - query_time))
        print('Total time: {}'.format(time.time() - total_time))
        print('Cardinality: {}'.format(sum(bucket_sizes)))

    return num_duplicates, cardinalities, duplicates_per_num_perm

def __main__():

    train_data = pd.read_csv('./Part-3/data/train_q_3.1.csv', names=['Question'],  delimiter='\t', header=0)
    test_data = pd.read_csv('./Part-3/data/test_without_labels_q_3.1.csv',  delimiter='\t')

    print("Train data shape: ", train_data.shape)
    print("\nTest data shape: ", test_data.shape)

    train_data['text'] = train_data['Question'].progress_apply(preprocess_text)
    test_data['text'] = test_data['Question'].progress_apply(preprocess_text)

    # count nan values
    test_data = test_data.dropna()

    vectorizer = CountVectorizer()
    X_train_count = vectorizer.fit_transform(train_data['text'])
    X_test_count = vectorizer.transform(test_data['text'])

    X_train = X_train_count
    X_test = X_test_count

    print("\n\n# Exact cosine similarity")
    exact_cosine(X_train, X_test)

    print("\n\n# Random projection LSH with Cosine similarity")
    cardinalities = []
    duplicates_per_k = []
    for k in range(1, 10+1):
        print("\n> k = ", k)
        num_duplicates, num_candidates_per_test_doc = random_projection_hashing(X_train, X_test, train_data, k)
        cardinalities.append(sum(num_candidates_per_test_doc))
        duplicates_per_k.append(num_duplicates)
    # Plot cardinalities
    plt.plot(range(1, 10+1), cardinalities)
    plt.xlabel('k')
    plt.ylabel('Cardinality')
    plt.title('Cardinality vs k')
    plt.show()
    
    # Plot duplicates per k
    plt.plot(range(1, 10+1), duplicates_per_k)
    plt.xlabel('k')
    plt.ylabel('Duplicates')
    plt.title('Duplicates vs k')
    plt.show()    
    
    print("\n\n# Exact Jaccard similarity - Vectors")
    exact_jaccard_vectors(X_train.toarray(), X_test.toarray())
    
    print("\n\n# Exact Jaccard similarity - Tokens")
    exact_jaccard_tokens(train_data['text'].tolist(), test_data['text'].tolist())

    print("\n\n# MinHash LSH with Jaccard similarity")
    num_permutations = [16, 32, 64]
    threshold = 0.8  # Similarity threshold for LSH
    duplicates, cardinalities, duplicates_per_num_perm = lsh_minhash(train_data['text'].tolist(), test_data['text'].tolist(), num_permutations, threshold)

    # Plot duplicates per num_perm
    plt.plot(num_permutations, duplicates_per_num_perm)
    plt.xlabel('Number of permutations')
    plt.ylabel('Duplicates')
    plt.title('Duplicates vs num_perm')
    plt.show()
    
    # Plot cardinalities per num_perm
    plt.plot(num_permutations, cardinalities)
    plt.xlabel('Number of permutations')
    plt.ylabel('Cardinality')
    plt.title('Cardinality vs num_perm')
    plt.show()
    
__main__()