from time import strptime
from turtle import title
import matplotlib.pyplot as plt
from numpy import percentile
import seaborn as sns
color = sns.color_palette()
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def data_read_clean (df):
    df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
    df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column
    return df


def pie_chart(df):     # Creating a pie chart with % to show counts with number of likes
    lables = df['Emotion']
    values = df['Number of Likes']
    fig = go.Figure(data=[go.Pie(labels=lables, values=values)])
    fig.update_layout(
    title_text="Percentage of emotion and likes") 
    fig.show()


def histo(df):
    # Creates a histogram based on the number of likes for each tweet
    fig = px.histogram(df, x="Emotion", text_auto = True) # Takes data from the column "Number of Likes"
    fig.update_traces(marker_color="blue", textfont_size = 20,
                    marker_line_width=1)
    fig.update_layout(title_text='Polarity of tweets')
    fig.show()


def wordcloud(tweet, title): # Creating a wordcloud with different emotions
    # image = np.array(Image.open('hashtag.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, 
                        background_color="white", stopwords=stopwords, min_font_size=10).generate(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('wordcloud10.png')
    plt.title(title, size = 20)
    plt.show()


df = data_read_clean(pd.read_csv('data.csv'))  # Read data from csv
# pie_chart(df)
# histo(df)

df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion
# wordcloud_p = wordcloud(df_positive, "Positive Word Cloud")
# wordcloud_n = wordcloud(df_negative, "Negative Word Cloud")


# Graph to show sentiments over time
df['Date Created'] = pd.to_datetime(df['Date Created'])
df['Date Created'] = df['Date Created'].dt.date
grouping = df.groupby(by='Date Created')['Emotion'].value_counts()
unstack_graph = grouping.unstack(level=1)
unstack_graph.plot.bar()
plt.show()