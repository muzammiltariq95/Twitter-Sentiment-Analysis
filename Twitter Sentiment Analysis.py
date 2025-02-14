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

X = dataset['text']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    
                                                    cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)random_state=42)

print("\n All data labels")
print(dataset.groupby('target').count())

resampler = RandomUnderSampler(random_state=0)
X_train, y_train = resampler.fit_resample(X_train, y_train)
sns.countplot(x=y_train)
plt.show()

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm,'\n\n')
print('--------------------------------------------------------')
result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n",)
print (result)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive','Neutral','Irrelevant'], 
            yticklabels=['Negative', 'Positive','Neutral','Irrelevant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def analyze_sentiment(review_text):
    # Preprocess the review text
    processed_text = preprocessing(review_text)
    vectorized_text = cv.transform([processed_text])
    sentiment = model.predict(vectorized_text)[0]
    return f"Sentiment for your review: {sentiment}"

analyze_sentiment('The service was excellent, very satisfied!')

analyze_sentiment('I absolutely love this product!')

analyze_sentiment("This is the worst experience I have ever had")

analyze_sentiment("The product broke after one use, awful!")

analyze_sentiment("It’s okay, nothing special to be honest.")

analyze_sentiment("Meh, could’ve been better, but not bad.")

analyze_sentiment("What’s the weather like today?")

analyze_sentiment("I’ll be visiting my friend later.")

pos_freq = FreqDist(positive_text)
pos_freq.tabulate(10)

neg_freq = FreqDist(negative_text)
neg_freq.tabulate(10)

pos_freq.plot(50)
plt.show()

neg_freq.plot(50)
plt.show()