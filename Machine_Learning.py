import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import seaborn as sns

# https://www.kaggle.com/code/sasakitetsuya/airlines-tweet-analysis-trial-by-word2vec-and-lstm
# https://www.analyticsvidhya.com/blog/2021/11/an-empirical-study-of-machine-learning-classifier-with-tweet-sentiment-classification/

def data_read_clean (df):
    df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
    df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column
    return df

df = data_read_clean(pd.read_csv('data.csv'))  # Read data from csv and drop duplicates from column "Tweet"

