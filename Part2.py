#!/usr/bin/env python
# coding: utf-8

# # Part 2: Nearest Neighbor Search with Locality Sensitive Hashing
# 
# 
# 
# Students:
# - Konstantinos Nikoletos 
# - Konstantinos Plas

# ## Question 2.1: Nearest Neighbor Search without and with Locality Sensitive Hashing
# 

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


# In[4]:


# stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = re.sub(r'\W', ' ', str(text))
    processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)
    processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)
    processed_text = re.sub(r'^b\s+', '', processed_text)

    tokens = [lemmatizer.lemmatize(word) for word in processed_text.split() if word not in stop_words]
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(tokens)

    return processed_text


# In[5]:


# from tqdm.notebook import tqdm
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

train_data = train_data.head(100)
test_data = test_data.head(100)

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from datasketch import MinHashLSH, MinHash
import numpy as np

# def jaccard_similarity(a, b):
#     intersection_size = len(set(a) & set(b))
#     union_size = len(set(a) | set(b))

#     return intersection_size / union_size if union_size > 0 else 0.0

# def jaccard_similarity(a, b):
    
#     a = a.toarray()
#     b = b.toarray()
    
#     set_a = set(a)
#     set_b = set(b)
    
#     intersection_size = len(np.intersect1d(a, b))
#     union_size = len(set_a) + len(set_b) - intersection_size
    
#     return intersection_size / union_size if union_size > 0 else 0.0

# def jaccard_similarity(set1, set2):
#     set1 = set1.toarray()
#     set2 = set2.toarray()
#     intersection_size = len(np.intersect1d(set1, set2))
#     union_size = len(np.union1d(set1, set2))
    
#     if union_size == 0:
#         return 0.0  # Jaccard similarity is 0 if the sets are both empty
#     else:
#         return intersection_size / union_size

from scipy.spatial.distance import jaccard

def jacc_sim(a, b):
    return 1-jaccard(a,b)


test_data_aslist = test_data['text'].tolist()
train_data_aslist = train_data['text'].tolist()

# Define parameters
k_neighbors = 15  # Number of neighbors for K-NN
threshold = 0.3  # Similarity threshold for LSH

# Create TF-IDF vectorizer
vectorizer = CountVectorizer(max_features=1024)
# vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(train_data['text'])
X_test_tfidf = vectorizer.transform(test_data['text'])

# Build MinHash LSH index
num_permutations = [16, 32, 64]

# Convert sparse matrices to dense arrays
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

print("Calculating KNN...")
start_knn_time = time.time()

# Check if the true KNN distances and indices have already been calculated
if os.path.exists('true_knn_distances.npy') and os.path.exists('true_knn_indices.npy'):
    print("Loading true KNN distances and indices from file...")
    true_knn_distances = np.load('true_knn_distances.npy')
    true_knn_indices = np.load('true_knn_indices.npy')
else:
    true_knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='brute', metric=jacc_sim).fit(X_train_dense)
    true_knn_distances, true_knn_indices = true_knn.kneighbors(X_test_dense)
    print("Finished calculating KNN.")
    print(f"KNN Time: {time.time() - start_knn_time:.4f} seconds")

    np.save('true_knn_distances.npy', true_knn_distances)
    np.save('true_knn_indices.npy', true_knn_indices)

def lsh_knn(candidates, train_set, test_doc):
    similarities = [(idx, jaccard_similarity(set(test_doc.split()), set(train_set[idx].split())))
                    for idx in candidates]

    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    k_most_similar = sorted_similarities[:k_neighbors]

    return k_most_similar, similarities

for num_perm in num_permutations:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    minhash_signatures_train = []
    for i, doc in enumerate(train_data_aslist):
        # print("doc: ", doc)
        minhash = MinHash(num_perm=num_perm)
        for word in set(doc.split()):
            # print("word: ", word)
            minhash.update(word.encode('utf8'))
        minhash_signatures_train.append(minhash)
    
    # print("minhash_signatures_train: ", len(minhash_signatures_train))
    for i, minhash in enumerate(minhash_signatures_train):
        # print("i: ", i, "minhash: ", minhash)
        lsh.insert(i, minhash)

    start_query_time = time.time()

    correct_predictions = 0
    total_fraction = 0
    lsh_indices = []
    lsh_distances = []
    avg_bucket_size = []
    for i, doc in enumerate(test_data_aslist):
        minhash = MinHash(num_perm=num_perm)
        # print("testdoc: ", doc)
        for word in set(doc.split()):
            # print("word: ", word)
            minhash.update(word.encode('utf8'))

        candidates = lsh.query(minhash)
        # print(candidates)
        similarities = true_knn_distances[i]
        true_indices = true_knn_indices[i]
        
        avg_bucket_size.append(len(candidates))
        num_of_true_docs = sum(1 for item in candidates if item in true_indices)
        total_fraction += (num_of_true_docs / true_indices.shape[0])
        # exit(1)
#         if candidates:
#             bucket_indices, bucket_distances = lsh_knn(candidates, train_data_aslist, doc)
#             lsh_indices.append(bucket_indices)
#             lsh_distances.append(bucket_distances)
    
    # print(f"Average Bucket Size: {sum(avg_bucket_size) / len(avg_bucket_size)}")
    # # plot histogram of bucket sizes
    # plt.hist(avg_bucket_size, bins=20)
    # plt.xlabel("Bucket Size")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Bucket Sizes")
    # plt.show()
    
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
    print(f"Fraction of True K-most similar documents: {accuracy:.4f}")

