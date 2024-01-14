import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
import time

from sklearn.neighbors import NearestNeighbors

# Data reading of train and test csv files
train_data = pd.read_csv('./data/train.csv', sep=',')
test_data = pd.read_csv('./data/test_without_labels.csv', sep=',')

train_data = train_data.head(1000)
test_data = test_data.head(1000)


# Combine Title and Content for each document
train_data['Combined'] = train_data['Title'] + ' ' + train_data['Content']
test_data['Combined'] = test_data['Title'] + ' ' + test_data['Content']

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
train_matrix = vectorizer.fit_transform(train_data['Combined'])
test_matrix = vectorizer.transform(test_data['Combined'])

# Split the train set for validation
train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Vectorize the validation set
validation_matrix = vectorizer.transform(validation_set['Combined'])

# Find the 15 nearest neighbors for each document in the validation set
print("Training the model...")
nn_model = NearestNeighbors(n_neighbors=15, metric='cosine', algorithm='brute')
nn_model.fit(train_matrix)

# Predict using the brute-force approach with NearestNeighbors
predict_start_time = time.time()
print("Predicting...")
distances, indices = nn_model.kneighbors(validation_matrix)

print(indices)
print(indices[:, 0])
print(distances)
print(distances[:, 0])
predictions = train_data.iloc[indices[:, 0]]['Label'].mode().values
predict_end_time = time.time()
print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")



k = 15
