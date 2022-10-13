import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import gensim
import pad_sequences
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import time
from keras.layers import Dense, Input, Flatten, LSTM, Bidirectional,Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential, load_model
from keras import losses
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping


df = pd.read_csv('data.csv')  # Read data from csv
#print(df['Tweet'])
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')


X = df.Tweet
y = df.Emotion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

def tfidf(words):
    tfidf_vectorizer = TfidfVectorizer()
    data_feature = tfidf_vectorizer.fit_transform(words)
    return data_feature, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train.tolist())
X_test_tfidf = tfidf_vectorizer.transform(X_test.tolist())

X_train_tfidf.shape

lr_tfidf = LogisticRegression(random_state=42,solver = 'liblinear')
lr_tfidf.fit(X_train_tfidf, y_train)
y_predicted_lr = lr_tfidf.predict(X_test_tfidf)

def score_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted,average= 'macro')
    recall = recall_score(y_test, y_predicted,average='macro')
    print("accuracy = %0.3f, precision = %0.3f, recall = %0.3f" % (accuracy, precision, recall))

score_metrics(y_test, y_predicted_lr)

def plot_confusion_matrix(y_test, y_predicted, title='Confusion Matrix'):
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True, fmt='.20g')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(y_test, y_predicted_lr)


w2v = Word2Vec(sentences = np.array(df).tolist(), window = 3, vector_size = 100, min_count = 5, workers = 4, sg = 1)
w2v.wv.most_similar('how')