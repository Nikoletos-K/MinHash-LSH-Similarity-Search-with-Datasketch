import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from .utils import preprocess_text


# Data reading of train and test csv files
train_data = pd.read_csv('./data/train.csv', sep=',')
test_data = pd.read_csv('./data/test_without_labels.csv', sep=',')

# train_data = train_data.head(1000)
print("Train data shape: ", train_data.shape)
print(train_data.head())
train_data['text'] = train_data['Title'] + " " + train_data['Content']

print("\nTest data shape: ", test_data.shape)
print(test_data.head())
test_data['text'] = test_data['Title'] + " " + test_data['Content']


enable_preprocess_text = True
# Define features and target
if enable_preprocess_text:
    print("Preprocessing text...")
    train_data['text'] = train_data['text'].apply(preprocess_text)
    test_data['text'] = test_data['text'].apply(preprocess_text)
    print("Preprocessing text done.")

X = train_data['text'].astype(str)
y = train_data['Label']

print(X)
print(y)
# Define the classifiers
svm_classifier = SVC()
rf_classifier = RandomForestClassifier()

# Define feature extraction methods
vectorizers = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer(),
}

svd = TruncatedSVD(n_components=100)

# Evaluate each classifier + feature combination
for classifier_name, classifier in [('SVM', svm_classifier), ('Random Forest', rf_classifier)]:
    for vectorizer_name, vectorizer in vectorizers.items():
        # Build the pipeline with feature extraction and classifier
        pipeline = make_pipeline(vectorizer, svd, classifier)

        # Perform 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Make predictions using cross-validation
        predictions = cross_val_predict(pipeline, X, y, cv=skf)

        # Calculate evaluation metrics
        accuracy = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy').mean()
        precision = cross_val_score(pipeline, X, y, cv=skf, scoring='precision_macro').mean()
        recall = cross_val_score(pipeline, X, y, cv=skf, scoring='recall_macro').mean()
        fscore = cross_val_score(pipeline, X, y, cv=skf, scoring='f1_macro').mean()
        # Print results
        print(f"\nResults for {classifier_name} + {vectorizer_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        
# Use your best model (e.g., SVM with BoW in this example)
best_model = make_pipeline(TfidfVectorizer(), svd, SVC(kernel='linear'))
best_model.fit(X, y)


X_test = test_data['text'].astype(str)
# Make predictions on the test set

print(X_test)
predictions = best_model.predict(X_test)

# Create a DataFrame with the results
output_df = pd.DataFrame({'Id': test_data['Id'], 'Predicted': predictions})

# sort by id ascending
output_df.sort_values(by=['Id'], inplace=True)

# Save the results to the output file
output_df.to_csv('testSet_categories.csv', index=False)

print("Predictions saved to testSet_categories.csv")