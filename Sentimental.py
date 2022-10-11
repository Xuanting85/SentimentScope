from turtle import title
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px


df = pd.read_csv('data.csv')  # Read data from csv #twtfile name
df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps

#Filter Positive & Negative words (eg, certain words are guarantee negative etc)
# df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column
print(len(df))

# Creates a histogram based on the number of likes for each tweet
fig = px.histogram(df, x="Emotion", text_auto = True) # Takes data from the column "Number of Likes"
fig.update_traces(marker_color=["green","blue","red"], textfont_size = 20,
                  marker_line_width=1)
fig.update_layout(title_text='Polarity of tweets')
fig.show()

df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion
df_sentiment = df['Emotion'] # want to classify the pos & neg as 1 grp

# Creating a wordcloud with different emotions
def wordcloud(tweet, title):
    # image = np.array(Image.open('hashtag.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, 
                        background_color="white", 
                        stopwords=["healthcare workers", "healthcare","never","so","before", "teachers"], #removing certain words from showing in wordcloud
                        colormap="Blues", min_font_size=10).generate(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('wordcloud10.png')
    plt.title(title, size = 20)
    plt.show()

# wordcloud_p = wordcloud(df_positive, "Positive Word Cloud")
# wordcloud_n = wordcloud(df_negative, "Negative Word Cloud")
# wordcloud_all = wordcloud(df_sentiment, "All Words Cloud")

# Graph to show sentiments over time
print(len(df))
neg = df[df['Emotion']=='negative']
neg = neg.groupby(['Date Created'],as_index=False).count()

pos = df[df['Emotion']=='positive']
pos = pos.groupby(['Date Created'],as_index=False).count()

pos = pos[['Date Created','id']]
neg = neg[['Date Created','id']]

#Pie Chart
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 



# df.value_counts().plot.pie()
df_sentiment.value_counts().plot.pie(y= df_sentiment, subplots=True, figsize =(5,5), colors = ['red', 'pink', 'purple'] )
plt.show()
