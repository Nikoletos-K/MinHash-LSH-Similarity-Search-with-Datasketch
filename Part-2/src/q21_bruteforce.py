import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import time

# Data reading of train and test csv files
train_data = pd.read_csv('./data/train.csv', sep=',')
test_data = pd.read_csv('./data/test_without_labels.csv', sep=',')


train_data = train_data.head(1000)


print("Train data shape: ", train_data.shape)
print(train_data.head())
train_data['text'] = train_data['Title'] + " " + train_data['Content']

print("\nTest data shape: ", test_data.shape)
print(test_data.head())
test_data['text'] = test_data['Title'] + " " + test_data['Content']

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
train_matrix = vectorizer.fit_transform(train_data['text'])
test_matrix = vectorizer.transform(test_data['text'])

# Split the train set for validation
train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Vectorize the validation set
validation_matrix = vectorizer.transform(validation_set['text'])

# Train the model using NearestNeighbors
print("Training the model...")
nn_model = NearestNeighbors(n_neighbors=15, metric='cosine', algorithm='brute')
nn_model.fit(train_matrix)

# Predict using the brute-force approach with NearestNeighbors
predict_start_time = time.time()
print("Predicting...")
distances, indices = nn_model.kneighbors(validation_matrix)
predictions = train_data.iloc[indices[:, 0]]['Label'].mode().values
predict_end_time = time.time()
print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Evaluate accuracy
accuracy = accuracy_score(validation_set['Label'], predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")