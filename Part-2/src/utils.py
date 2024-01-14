import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Preprocess data
def preprocess_text(text):
    # Remove all the special characters
    processed_text = re.sub(r'\W', ' ', str(text))

    # Remove all single characters
    processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)

    # Remove single characters from the start
    processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)

    # Substituting multiple spaces with single space
    processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)

    # Removing prefixed 'b'
    processed_text = re.sub(r'^b\s+', '', processed_text)

    # Converting to Lowercase
    processed_text = processed_text.lower()
    
    # Remove stopwords
    # processed_text = processed_text.split()
    stop_words = set(stopwords.words('english'))
    # processed_text = [word for word in processed_text if not word in stop_words]
    # processed_text = ' '.join(processed_text)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    processed_text = processed_text.split()

    processed_text = [lemmatizer.lemmatize(word) for word in processed_text if not word in stop_words]
    processed_text = ' '.join(processed_text)

    return processed_text