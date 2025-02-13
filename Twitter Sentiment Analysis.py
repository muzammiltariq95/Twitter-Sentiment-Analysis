# !pip install numpy pandas matplotlib seaborn wordcloud nltk scikit-learn imbalanced-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import re
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
from nltk.probability import FreqDist
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'vader_lexicon'])

dataset = pd.read_csv('twitter_training.csv', header=None)

dataset.head()

dataset.info()

dataset.drop(columns= [0,1], inplace=True)

dataset.dropna(inplace=True)

dataset.columns= ["target", "text"]

dataset.isnull().sum()

dataset['text']=dataset['text'].str.lower()
dataset.head()

print("Data label count")
print(dataset.groupby("target").count())

dataset['target'].value_counts().plot(kind='bar', 
                                      title='Class Distribution', 
                                      xlabel='Sentiment', ylabel='Count')
plt.show()

all_words = ' '.join(dataset['text']).split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(10)
print("Most Common Words:", most_common_words)

dataset['text_length'] = dataset['text'].apply(len)
print(dataset['text_length'].describe())
dataset['text_length'].hist(bins=20)
plt.title("Text Length Distribution")
plt.xlabel("Length of Text")
plt.ylabel("Frequency")
plt.show()

tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+')

stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.PorterStemmer()

def preprocessing(text):
    tokenized = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text)
    cleaned_tokens =  [word.lower() for word in tokenized if word.lower() not in stopwords]
    stemmed_text = [nltk.stem.PorterStemmer().stem(word) for word in cleaned_tokens]
    stemmed_text = ' '.join(stemmed_text)
    return stemmed_text

dataset['text'] = dataset['text'].apply(preprocessing)

# Combine all text in the dataset into a single string
all_text = ' '.join(dataset['text'])

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.title("Word Cloud for Dataset", fontsize=16)
plt.show()

wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    stopwords=STOPWORDS
).generate(all_text)

positive_text = ' '.join(dataset[dataset['target'] == 'Positive']['text'])
negative_text = ' '.join(dataset[dataset['target'] == 'Negative']['text'])

# Word cloud for positive sentiment
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Positive Sentiment", fontsize=16)
plt.show()

# Word cloud for negative sentiment
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Negative Sentiment", fontsize=16)
plt.show()

