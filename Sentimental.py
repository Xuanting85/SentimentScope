from datetime import datetime
from time import strptime
from tracemalloc import start
from turtle import title
import matplotlib.pyplot as plt
from numpy import percentile
from regex import D
import seaborn as sns
color = sns.color_palette()
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def data_read_clean(df):
    df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
    df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column
    remove_words = ["healthcare", "Healthcare", "Worker", "worker", "healthcare worker", "workers", "never", "so", "before", 
    "healthcare workers"] # Specify common words to be removed
    rem = r'\b(?:{})\b'.format('|'.join(remove_words)) # Set parameters to remove this list of words from "Tweet" column
    df['Tweet'].str.replace(rem, '') 
    return df

# Functions below perform visualization with different charts / graphs

def pie_chart(df):     # Creates a pie chart to count % of each emotion
    emotions = df['Emotion'].value_counts()
    emotions.plot.pie(y=emotions, subplots=True, figsize=(5,5),colors=['green','red','blue'], autopct='%1.0f%%')
    plt.show()

    
def histo(df): # Creates a histogram based on the number of likes for each tweet
    fig = px.histogram(df, x="Emotion", y ="Number of Likes", title="Number of likes for each emotion", width=1200, height=1000) # Takes data from the column "Number of Likes"
    fig.update_traces(textfont_size = 100,
                    marker_line_width=1, marker_color=["blue", "red", "green"])
    fig.show()


def wordcloud(tweet, title, col): # Creating a wordcloud with different emotions
    # image = np.array(Image.open('hashtag.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, 
                        background_color="white", stopwords=stopwords, min_font_size=15, colormap=col).generate(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, size = 20)
    plt.show()


def kernal_graph(df):# Kernal distribution graph
# Estimate density of the distribution of emotion / similar to a histogram
    num_neg = df[df['Emotion']=='Negative']['Tweet'].apply(lambda x: len(x.split()))
    num_neu = df[df['Emotion']=='Neutral']['Tweet'].apply(lambda x: len(x.split()))
    num_pos = df[df['Emotion']=='Positive']['Tweet'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12,6))
    sns.kdeplot(num_neg, shade=True, color = 'r').set_title('Distribution of number of words')
    sns.kdeplot(num_neu, shade=True, color = 'y')
    sns.kdeplot(num_pos, shade=True, color = 'b')

    plt.legend(labels=['Negative', 'Neutral','Positive'])
    plt.show()


def time_bar(start, end, df): # Graph to show sentiments over time
    df['Date Created'] = pd.to_datetime(df['Date Created'])
    df['Date Created'] = df['Date Created'].dt.date
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    mask = (df['Date Created'] > start) & (df['Date Created'] <= end)
    df2 = df.loc[mask]
    grouping = df2.groupby(by='Date Created')['Emotion'].value_counts()
    unstack_graph = grouping.unstack(level=1)
    unstack_graph.plot.bar()
    plt.show()


df = data_read_clean(pd.read_csv('data.csv'))  # Read data from csv and drop duplicates from column "Tweet"


# pie_chart(df)

# emotion_count = df['Emotion'].value_counts() # Count the occurence of each emotion
# counts = pd.DataFrame({'FuncGroup' :emotion_count.index, 'Count':emotion_count.values})
# print(counts) 

histo(df)

# kernal_graph(df)

df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion
def_neutral = df.loc[df['Emotion'] == "Neutral"] # Selecting columns with neutral emotion
# wordcloud_p = wordcloud(df_positive, "Positive Word Cloud", "Greens")
# wordcloud_n = wordcloud(df_negative, "Negative Word Cloud", "Reds")
# wordcloud_neu = wordcloud(def_neutral, "Neutral Word Cloud", "Blues")

# time_bar('2019-11-12','2019-11-21', df)