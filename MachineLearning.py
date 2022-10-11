import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('data.csv')  # Read data from csv
#print(df['Tweet']) 
#function to split text into word
for i in df['Tweet']:
    tokens = word_tokenize(i)
stop_words=stopwords.words('english')
stemmer=PorterStemmer()
cleaned_data=[]
for i in range(len(df['Tweet'])):
   tweet=re.sub('[^a-zA-Z]',' ',df['Tweet'].iloc[i])
   tweet=tweet.lower().split()
tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)]
tweet=' '.join(tweet)
cleaned_data.append(tweet)
cv=CountVectorizer(max_features=3000,stop_words=['virginamerica','unit'])
X_fin=cv.fit_transform(cleaned_data).toarray()
y = df['Tweet']
sentiment_ordering = ['negative', 'neutral', 'positive']
y = y.apply(lambda x: sentiment_ordering.index(x))
model = MultinomialNB()
X_train,X_test,y_train,y_test = train_test_split(X_fin,y,test_size=0.9)
model.fit(X_train,y_train)