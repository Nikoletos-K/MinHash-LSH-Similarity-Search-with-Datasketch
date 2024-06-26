#!/usr/bin/env python
# coding: utf-8

# Part 2: Nearest Neighbor Search with Locality Sensitive Hashing
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

from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

train_data = pd.read_csv('./Part-1/data/train.csv', sep=',')
test_data = pd.read_csv('./Part-1/data/test_without_labels.csv', sep=',')

print("Train data shape: ", train_data.shape)
print(train_data.head())
train_data['text'] = train_data['Title'] + " " + train_data['Content']

print("\nTest data shape: ", test_data.shape)
print(test_data.head())
test_data['text'] = test_data['Title'] + " " + test_data['Content']


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

preprocessed_file_path_train = 'pre_train.csv'
if not os.path.exists(preprocessed_file_path_train):
    print("[TRAIN] Preprocessing text...")
    train_data['text'] = train_data['text'].progress_apply(preprocess_text)
    print("[TRAIN] Preprocessing text done.")
    train_data.to_csv(preprocessed_file_path_train, index=False)
else:
    print("[TRAIN] Reading from file")
    train_data = pd.read_csv(preprocessed_file_path_train)

preprocessed_file_path_test = 'pre_test.csv'
if not os.path.exists(preprocessed_file_path_test):
    print("[TEST] Preprocessing text...")
    test_data['text'] = test_data['text'].progress_apply(preprocess_text)
    print("[TEST] Preprocessing text done.")
    test_data.to_csv(preprocessed_file_path_test, index=False)
else:
    print("[TEST] Reading from file")
    test_data = pd.read_csv(preprocessed_file_path_test)

# train_data = train_data.head(100)
# test_data = test_data.head(100)

# train_data['text'] = train_data['text'].progress_apply(preprocess_text)
# test_data['text'] = test_data['text'].progress_apply(preprocess_text)

# LSH implementation

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from datasketch import MinHashLSH, MinHash
import numpy as np
from scipy.spatial.distance import jaccard
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def jacc_sim(a, b):
    return 1-jaccard(a,b)

test_data_aslist = test_data['text'].tolist()
train_data_aslist = train_data['text'].tolist()

# Parameters
k_neighbors = 15  # Number of neighbors for K-NN
threshold = 0.8  # Similarity threshold for LSH

# Create vectorizer
vectorizer = CountVectorizer(max_features=512)

X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Build MinHash LSH index
num_permutations = [16, 32, 64]

# Convert sparse matrices to dense arrays
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

scaler = StandardScaler()
X_train_dense = scaler.fit_transform(X_train_dense)
X_test_dense = scaler.transform(X_test_dense)

print("Calculating KNN...")
start_knn_time = time.time()

# If the true KNN distances and indices have already been calculated
if os.path.exists('true_knn_distances.npy') and os.path.exists('true_knn_indices.npy'):
    print("Loading true KNN distances and indices from file...")
    true_knn_distances = np.load('true_knn_distances.npy')
    true_knn_indices = np.load('true_knn_indices.npy')
    print("Finished loading true KNN distances and indices from file.")
else:
    true_knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='brute', metric=jaccard, n_jobs=4).fit(X_train_dense)
    true_knn_distances, true_knn_indices = true_knn.kneighbors(X_test_dense)
    print("Finished calculating KNN.")
    print(f"KNN Time: {time.time() - start_knn_time:.4f} seconds")

    np.save('true_knn_distances.npy', true_knn_distances)
    np.save('true_knn_indices.npy', true_knn_indices)

# Heatmap for true KNN distances seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(true_knn_distances)
plt.xlabel("KNN Index")
plt.ylabel("Test Data Index")
plt.title("Heatmap of True KNN Distances")
plt.savefig('heatmap.png', bbox_inches='tight')
plt.clf()

#  Histogram of true KNN distances
plt.hist(true_knn_distances.flatten(), bins=20)
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Histogram of True KNN Distances")
plt.savefig('histogram.png', bbox_inches='tight')
plt.clf()

count_greater_than_threshold = np.sum(true_knn_distances < 1-threshold)
print(f"Number of pairs with similarity > {threshold}: {count_greater_than_threshold}")
print("Calculating LSH...")
for num_perm in num_permutations:
    print(f"Calculating LSH for num_perm={num_perm} and threshold={threshold}...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    minhash_signatures_train = []
    print("Building LSH...")
    for i, doc in tqdm(enumerate(train_data_aslist)):
        minhash = MinHash(num_perm=num_perm)
        for word in set(doc.split()):
            minhash.update(word.encode('utf8'))
        minhash_signatures_train.append(minhash)

    for i, minhash in enumerate(minhash_signatures_train):
        lsh.insert(i, minhash)
    print("Finished building LSH.")

    start_query_time = time.time()

    correct_predictions = 0
    total_fraction = 0
    lsh_indices = []
    lsh_distances = []
    avg_bucket_size = []
    print("Quering for each doc file...")
    for i, doc in tqdm(enumerate(test_data_aslist)):
        minhash = MinHash(num_perm=num_perm)
        for word in set(doc.split()):
            minhash.update(word.encode('utf8'))

        candidates = lsh.query(minhash)
        similarities = true_knn_distances[i]
        true_indices = true_knn_indices[i]
        distance_threshold = 1-threshold
        true_indices = [true_knn_indices[i][j] for j in range(15) if similarities[j] < distance_threshold]

        avg_bucket_size.append(len(candidates))

        if len(candidates)>0 and len(true_indices)>0:
            num_of_true_docs = sum(1 for item in candidates if item in true_indices)
            total_fraction += (num_of_true_docs / len(true_indices))
        elif len(true_indices)==0:
            total_fraction += 1

    plt.hist(avg_bucket_size, bins=20)
    plt.xlabel("Bucket Size")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bucket Sizes")
    plt.savefig('buckets_'+str(num_perm)+'.png', bbox_inches='tight')
    plt.clf()

    end_query_time = time.time()
    build_time = time.time()
    query_time = end_query_time - start_query_time
    total_time = build_time + query_time

    print(f"\nResults for num_perm={num_perm}:")
    print(f"LSH Index Creation Time (BuildTime): {build_time:.4f} seconds")
    print(f"Total Query Time (QueryTime): {query_time:.4f} seconds")
    print(f"Total Time (TotalTime): {total_time:.4f} seconds")
    accuracy = total_fraction / len(test_data_aslist)
    print(f"Average Bucket Size: {sum(avg_bucket_size) / len(avg_bucket_size)}")
    print(f"> Fraction of True K-most similar documents: {accuracy:.4f}")
    print("\n\n")
