from turtle import title
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image

df = pd.read_csv('data.csv')  # Read data from csv
df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
print(len(df))

# Creates a histogram based on the number of likes for each tweet
fig = px.histogram(df, x="Emotion", text_auto = True) # Takes data from the column "Number of Likes"
fig.update_traces(marker_color="blue", textfont_size = 20,
                  marker_line_width=1)
fig.update_layout(title_text='Polarity of tweets')
fig.show()


df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion

# Creating a wordcloud with different emotions
def wordcloud(tweet, title):
    image = np.array(Image.open('hashtag.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, mask=image,
                        background_color="white", stopwords=stopwords, min_font_size=10).generate(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('wordcloud10.png')
    plt.title(title, size = 20)
    plt.show()

wordcloud_p = wordcloud(df_positive, "Positive Word Cloud")
wordcloud_n = wordcloud(df_negative, "Negative Word Cloud")


# Graph to show sentiments over time
print(len(df))
neg = df[df['Emotion']=='negative']
neg = neg.groupby(['Date Created'],as_index=False).count()

pos = df[df['Emotion']=='positive']
pos = pos.groupby(['Date Created'],as_index=False).count()

pos = pos[['Date Created','id']]
neg = neg[['Date Created','id']]
