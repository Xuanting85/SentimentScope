import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import seaborn as sns

# https://www.kaggle.com/code/sasakitetsuya/airlines-tweet-analysis-trial-by-word2vec-and-lstm
# https://www.analyticsvidhya.com/blog/2021/11/an-empirical-study-of-machine-learning-classifier-with-tweet-sentiment-classification/

df = pd.read_csv('data.csv')  # Read data from csv
df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
# df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column
print(df.head(5))
print(df.info())

# df.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()


# le_encoder = preprocessing.LabelEncoder()
# le_encoder.fit(df["Emotion"])

num_neg = df[df['Emotion']=='Negative']['Tweet'].apply(lambda x: len(x.split()))
num_neu = df[df['Emotion']=='Neutral']['Tweet'].apply(lambda x: len(x.split()))
num_pos = df[df['Emotion']=='Positive']['Tweet'].apply(lambda x: len(x.split()))
plt.figure(figsize=(12,6))
sns.kdeplot(num_neg, shade=True, color = 'r').set_title('Distribution of number of words')
sns.kdeplot(num_neu, shade=True, color = 'y')
sns.kdeplot(num_pos, shade=True, color = 'b')

plt.legend(labels=['Negative', 'Neutral','Positive'])
plt.show()