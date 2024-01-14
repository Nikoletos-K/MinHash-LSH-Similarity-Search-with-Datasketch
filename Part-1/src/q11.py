import pandas as pd
from tqdm import tqdm

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

# Text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Creating a wordcloud for each category of news
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

def generate_wordcloud(text, title="WordCloud", remove_stopwords=True):
    """
    Generate a word cloud from the given text after removing stopwords.

    Parameters:
    - text (str): Input text for word cloud generation.
    - title (str): Title for the word cloud plot.

    Returns:
    - None
    """
    # Tokenize the text
    print("Tokenizing the text...")
    words = word_tokenize(text)

    # Remove stopwords
    if remove_stopwords:
        print("Removing stopwords...")
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in tqdm(words) if word.lower() not in stop_words]
    else:
        filtered_words = words
    # Join the filtered words back into a string
    print("Joining the filtered words back into a string...")
    filtered_text = ' '.join(filtered_words)

    # Generate the word cloud
    print("Generating the word cloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

    # Plot the WordCloud image
    print("Plotting the word cloud...")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)

    print("Saving the word cloud...")
    plt.savefig(title+'.png', bbox_inches='tight')
    # plt.show()

# Example usage:
text = " ".join(review for review in train_data[train_data['Label']=='Entertainment']['text'])
generate_wordcloud(text, title="Entertainment")
print("Entertainment Word cloud generated.")

text = " ".join(review for review in train_data[train_data['Label']=='Technology']['text'])
generate_wordcloud(text, title="Technology")
print("Technology Word cloud generated.")

text = " ".join(review for review in train_data[train_data['Label']=='Health']['text'])
generate_wordcloud(text, title="Health")
print("Health Word cloud generated.")

text = " ".join(review for review in train_data[train_data['Label']=='Business']['text'])
generate_wordcloud(text, title="Business")
print("Business Word cloud generated.")