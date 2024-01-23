#!/usr/bin/env python
# coding: utf-8

# # Part 1: Text classification
# 
# Students:
# - Konstantinos Nikoletos
# - Konstantinos Plas

# ## Question 1.1: Get to know the Data: WordCloud

# In[1]:


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


# ## Reading the dataset

# In[2]:


train_data = pd.read_csv('./Part-1/data/train.csv', sep=',')
test_data = pd.read_csv('./Part-1/data/test_without_labels.csv', sep=',')


print("Train data shape: ", train_data.shape)
print(train_data.head())
train_data['text'] = train_data['Title'] + " " + train_data['Content']

print("\nTest data shape: ", test_data.shape)
print(test_data.head())
test_data['text'] = test_data['Title'] + " " + test_data['Content']


# ## WordCloud Source code

# In[3]:


def generate_wordcloud(text, title="WordCloud"):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    print("Plotting the word cloud...")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)

    print("Saving the word cloud...")
    plt.savefig(title+'.png', bbox_inches='tight')


# ### Entertainment WordCloud

# In[4]:


text = " ".join(review for review in train_data[train_data['Label']=='Entertainment']['text'])
generate_wordcloud(text, title="Entertainment")
print("Entertainment Word cloud generated.")


# ### Technology WordCloud

# In[5]:


text = " ".join(review for review in train_data[train_data['Label']=='Technology']['text'])
generate_wordcloud(text, title="Technology")
print("Technology Word cloud generated.")


# ### Health WordCloud

# In[6]:


text = " ".join(review for review in train_data[train_data['Label']=='Health']['text'])
generate_wordcloud(text, title="Health")
print("Health Word cloud generated.")


# ### Business WordCloud

# In[7]:


text = " ".join(review for review in train_data[train_data['Label']=='Business']['text'])
generate_wordcloud(text, title="Business")
print("Business Word cloud generated.")


# ## Printing again WordClouds but having processed the text

# In[8]:


from stop_words import get_stop_words
stop_words_pypi = set(get_stop_words('en'))
# print(stop_words_pypi)

from nltk.corpus import stopwords
stop_words_nltk = set(stopwords.words('english'))
# print(stop_words_nltk)

manual_stop_words = {'include', 'way', 'work', 'look', 'add', 'time', 'year', 'one', \
                     'month', 'day', 'help', 'think', 'tell', 'new', 'said', 'say',\
                     'need', 'come', 'good', 'set', 'want', 'people', 'use', 'day', 'week', 'know'}

stop_words= stop_words_nltk.union(stop_words_pypi)
stop_words = stop_words.union(manual_stop_words)


# In[9]:


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


# In[12]:


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


# ### Entertainment WordCloud

# In[13]:


text = " ".join(review for review in train_data[train_data['Label']=='Entertainment']['text'])
generate_wordcloud(text, title="Entertainment")
print("Entertainment Word cloud generated.")


# ### Technology WordCloud

# In[14]:


text = " ".join(review for review in train_data[train_data['Label']=='Technology']['text'])
generate_wordcloud(text, title="Technology")
print("Technology Word cloud generated.")


# ### Health WordCloud

# In[15]:


text = " ".join(review for review in train_data[train_data['Label']=='Health']['text'])
generate_wordcloud(text, title="Health")
print("Health Word cloud generated.")


# ### Business WordCloud

# In[16]:


text = " ".join(review for review in train_data[train_data['Label']=='Business']['text'])
generate_wordcloud(text, title="Business")
print("Business Word cloud generated.")


# ## Question 1.2: Classification Task

# In[17]:


train_data = train_data.head(1000)
test_data = test_data.head(1000)


# ### BoW, Tf-Idf vectorization with K-fold for SVM and RandomForest

# In[18]:


from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np


# In[19]:


X = train_data['text'].astype(str)
y = train_data['Label']

vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(train_data['text'])
X_test_bow = vectorizer.transform(test_data['text'])

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_data['text'])
X_test_tfidf = vectorizer.transform(test_data['text'])


# In[20]:


svm_classifier = SVC(kernel='linear')
rf_classifier = RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=-1)
vectorizers = ['TF-IDF', 'BoW']


# In[21]:


for classifier_name, classifier in [('SVM', svm_classifier), ('Random Forest', rf_classifier)]:
    for vectorizer in vectorizers:

        X = X_train_tfidf if vectorizer == 'TF-IDF' else X_train_bow

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, _ in stratified_kfold.split(X_train, y_train):
            print(classifier_name, " with ", vectorizer, "[ ",  train_index[0], ",", train_index[-1] , " ]" )
            X_train_fold, y_train_fold = X_train[train_index], np.array(y_train)[train_index]
            classifier.fit(X_train_fold, y_train_fold)
        predictions = classifier.predict(X_test)

        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n\nResults for {classifier_name} + {vectorizer}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-Score: {f1:.4f}")
        print(classification_report(y_test, predictions))


# ### Best model - LinearSVC with Tf-Idf

# In[22]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


train_data = train_data.head(2000)
# test_data = test_data.head(1000)


# In[24]:


label_encoder = LabelEncoder()
train_data['_Label'] = label_encoder.fit_transform(train_data['Label'])

X = train_data['text']
y = train_data['_Label']

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_data['text'])
X_test_tfidf = vectorizer.transform(test_data['text'])

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn.svm import LinearSVC

classifier = LinearSVC(random_state=42, tol=1e-5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')
accuracy = accuracy_score(y_test, predictions)
print(f"\n\nResults for XGBClassifier:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-Score: {f1:.4f}")
print(classification_report(y_test, predictions))


# In[26]:


best_model = classifier
test_predictions = best_model.predict(X_test_tfidf)

output_df = pd.DataFrame({'Id': test_data['Id'], 'Predicted': label_encoder.inverse_transform(test_predictions)})
output_df.to_csv('testSet_categories.csv', index=False)

print("Predictions saved to testSet_categories.csv")

